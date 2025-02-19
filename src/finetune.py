import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing.spawn
import torch.nn as nn
import torch.optim as optim
import wandb

from args import Args
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from distributed import cleanup_ddp, distribute_loader, setup_ddp
from eval import evaluate
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from task_vectors import TaskVector
from utils import cosine_lr, LabelSmoothing


@torch.no_grad()
def calculate_norm(image_encoder: ImageEncoder) -> torch.Tensor:
    total_norm = 0.0
    count = 0
    state_dict: Dict[str, torch.Tensor] = image_encoder.model.state_dict()
    for name, param in state_dict.items():
        if "Delta.D" in name:
            total_norm += param.norm(p="fro")
            count += 1
        elif "Delta.A" in name:
            A = param
            B = state_dict[name.replace("A", "B")]
            total_norm += (B @ A).norm(p="fro")
            count += 1
    return total_norm / count


def finetune(rank: int, args: Args) -> ImageEncoder:
    """Finetune a pretrained model on a single dataset."""
    if args.wandb:
        config_dict = vars(args)
        run = wandb.init(
            project="Task Arithmetic",
            group=f"finetune_{args.finetuning_type}",
            name=f"{args.model_architecture}_{args.pretrained}_{args.train_dataset}_{args.seed}",
            config=config_dict,
            settings=wandb.Settings(start_method="thread")
        )

    print("=" * 100)
    print(f"Finetuning {args.model_architecture} on {args.train_dataset} with {args.finetuning_type} finetuning")
    print("=" * 100)
    print(f"Using Device: {args.device}")

    setup_ddp(rank, args.world_size, args.port)

    model_dir = os.path.join(
        args.model_root,
        args.model_architecture,
        args.pretrained,
        args.finetuning_type
    )

    # Load pretrained model
    if args.pretrained_model_path is not None:
        print(f"Loading a pretrained encoder from {args.pretrained_model_path}")
        image_encoder = ImageEncoder.load(args.pretrained_model_path)
    else:
        print(f"Building a {args.pretrained} pretrained encoder")
        image_encoder = ImageEncoder(args, keep_lang=False)
        args.pretrained_model_path = os.path.join(
            model_dir,
            f"zeroshot_rank_{args.rank}.pt"
        )
        image_encoder.save(args.pretrained_model_path)
    image_encoder.freeze_pretrained_weight(finetuning_type=args.finetuning_type)

    classification_head = get_classification_head(args, args.train_dataset)

    # Create a classifier
    classifier = ImageClassifier(image_encoder, classification_head)
    classifier.freeze_head()
    classifier = classifier.to(rank)

    print("\nTrainable Parameters:")
    for name, param in classifier.named_parameters():
        if param.requires_grad:
            print(name)

    preprocess_fn = classifier.image_encoder.train_preprocess

    # Load Dataset and DataLoader
    train_dataset = get_dataset(
        args.train_dataset+"Val", preprocess_fn, args.dataset_root,
        args.batch_size, args.num_workers
    )
    val_dataset = get_dataset(
        args.train_dataset+"Val", preprocess_fn, args.dataset_root,
        args.batch_size, args.num_workers
    )
    train_dataloader = get_dataloader(
        train_dataset, is_train=True, args=args, image_encoder=None
    )
    val_dataloader = get_dataloader(
        val_dataset, is_train=False, args=args, image_encoder=None
    )
    train_num_batches = len(train_dataloader)
    val_num_batches = len(val_dataloader)

    # Distribute
    ddp_train_loader = distribute_loader(train_dataloader)
    ddp_val_loader = distribute_loader(val_dataloader)
    ddp_classifier = torch.nn.parallel.DistributedDataParallel(
        classifier, device_ids=[rank],
        find_unused_parameters=False, output_device=rank
    )

    # Watch the model
    if args.wandb:
        wandb.watch(ddp_classifier, log="all", log_freq=250)

    # Define loss function, optimizer, and scheduler
    if args.ls > 0.0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = nn.CrossEntropyLoss()

    trainable_params = [p for p in ddp_classifier.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found"

    optimizer = optim.AdamW(
        params=trainable_params, lr=args.lr, weight_decay=args.wd
    )
    scheduler = cosine_lr(
        optimizer, args.lr, args.warmup_length,
        args.epochs * train_num_batches // args.grad_accum_steps
    )

    # Training loop
    print("\nStart Training\n")
    total_train_time = 0.0
    total_val_time = 0.0
    for epoch in range(args.epochs):
        train_losses = []
        train_accs = []
        training_batch_times = []
        val_losses = []
        val_accs = []
        validation_batch_times = []
        task_vector_norms = []
        learning_rates = []

        # Training
        epoch_train_start_time = time.time()
        ddp_classifier.train()
        ddp_train_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(ddp_train_loader):
            batch_start_time = time.time()
            step = i // args.grad_accum_steps + epoch * train_num_batches // args.grad_accum_steps
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(rank)
            labels = batch["labels"].to(rank)

            logits = ddp_classifier(inputs)
            predictions = torch.argmax(logits, dim=1)

            train_loss = loss_fn(logits, labels)
            train_loss.backward()

            train_corrects = (predictions == labels).sum().item()
            train_acc = train_corrects / len(labels)

            norm = calculate_norm(ddp_classifier.module.image_encoder)

            train_losses.append(train_loss.item())
            train_accs.append(train_acc)
            task_vector_norms.append(norm.item())
            learning_rates.append(optimizer.param_groups[0]["lr"])

            if (i + 1) % args.grad_accum_steps == 0:
                scheduler(step)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                training_batch_time = time.time() - batch_start_time
                training_batch_times.append(training_batch_time)
                percent_complete = 100 * i / len(ddp_train_loader)
                print(
                    f"Training on {args.train_dataset}\t Epoch: {epoch + 1}/{args.epochs}\t"
                    f"Batch: {i + 1}/{len(ddp_train_loader)} "
                    f"({percent_complete:.2f}%)\t"
                    f"Step: {step + 1}/{args.epochs * train_num_batches // args.grad_accum_steps}\t"
                    f"Training Loss: {train_loss.item():.4f}\t"
                    f"Training Accuracy: {train_acc:.2%}\t"
                    f"Task Vector Norm: {norm:.4f}\t"
                    f"lr: {optimizer.param_groups[0]['lr']:.8f}\t"
                    f"Training Batch Time: {training_batch_time:.2f}s", flush=True
                )

        total_train_time += time.time() - epoch_train_start_time
        print(f"\nEpoch: {epoch + 1} Training Time: {total_train_time:.2f}s\n")

        # Validation
        epoch_val_start_time = time.time()
        ddp_classifier.eval()
        ddp_val_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            for i, batch in enumerate(ddp_val_loader):
                step = i // args.grad_accum_steps + epoch * val_num_batches // args.grad_accum_steps
                batch_start_time = time.time()
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].to(rank)
                labels = batch["labels"].to(rank)

                logits = ddp_classifier(inputs)
                predictions = torch.argmax(logits, dim=1)

                val_loss = loss_fn(logits, labels)

                val_corrects = (predictions == labels).sum().item()
                val_acc = val_corrects / len(labels)

                val_losses.append(val_loss.item())
                val_accs.append(val_acc)

                if (i + 1) % args.grad_accum_steps == 0:
                    validation_batch_time = time.time() - batch_start_time
                    validation_batch_times.append(validation_batch_time)
                    percent_complete = 100 * i / len(ddp_val_loader)
                    print(
                        f"Validation on {args.train_dataset}\t Epoch: {epoch + 1}/{args.epochs}\t"
                        f"Batch: {i + 1}/{len(ddp_val_loader)} "
                        f"({percent_complete:.2f}%)\t"
                        f"Step: {step + 1}/{args.epochs * val_num_batches // args.grad_accum_steps}\t"
                        f"Validation Loss: {val_loss.item():.4f}\t"
                        f"Validation Accuracy: {val_acc:.2%}\t"
                        f"Validation Batch Time: {validation_batch_time:.2f}s", flush=True
                    )

            if args.save:
                if step % 200 == 0:
                    filename = os.path.join(
                        model_dir,
                        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
                        f"rank_{args.rank}_alpha_{args.alpha}",
                        "finetune",
                        f"bs_{args.batch_size}_seed_{args.seed}",
                        f"finetuned_task_vector_on_{args.train_dataset}_step_{step}.pt"
                    )
                    task_vector = TaskVector(
                        pretrained_checkpoint=ImageEncoder(args, keep_lang=False),
                        finetuned_checkpoint=ddp_classifier.module.image_encoder
                    )
                    task_vector.save_vector(filename)

        total_val_time += time.time() - epoch_val_start_time
        print(f"\nEpoch: {epoch + 1} Validation Time: {total_val_time:.2f}s\n")

        if args.wandb:
            run.log({
                "Epoch": epoch + 1,
                "Training Loss": np.mean(train_losses),
                "Training Accuracy": np.mean(train_accs),
                "Validation Loss": np.mean(val_losses),
                "Validation Accuracy": np.mean(val_accs),
                "Task Vector Norm": np.mean(task_vector_norms),
                "Learning Rate": np.mean(learning_rates),
                "Training Batch Time": np.mean(training_batch_times),
                "Validation Batch Time": np.mean(validation_batch_times)
            })

    print(f"Completed Training in {total_train_time:.2f}s")
    print(f"Completed Validation in {total_val_time:.2f}s")
    if args.wandb:
        run.log({
            "Total Training Time": total_train_time,
            "Total Validation Time": total_val_time
        })

    # Save the finetuned model
    if args.save:
        filename = os.path.join(
            model_dir,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"finetuned_image_encoder_on_{args.train_dataset}.pt"
        )
        ddp_classifier.module.image_encoder.save(filename)
        task_vector = TaskVector(
            pretrained_checkpoint=ImageEncoder(args, keep_lang=False),
            finetuned_checkpoint=ddp_classifier.module.image_encoder
        )
        filename = os.path.join(
            model_dir,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"finetuned_task_vector_on_{args.train_dataset}.pt"
        )
        task_vector.save_vector(filename)

    # Evaluate the finetuned model
    evaluate(ddp_classifier.module.image_encoder, args)

    cleanup_ddp()
    if args.wandb:
        run.finish()

    return ddp_classifier


if __name__ == "__main__":
    args: Args = Args().from_args()

    assert args.batch_size % args.grad_accum_steps == 0, \
        "Batch size must be divisible by gradient accumulation steps"
    args.batch_size = args.batch_size // args.grad_accum_steps

    assert args.finetuning_type in ["full", "linear", "lora"], \
        "Finetuning type must be one of: full, linear, lora"
    if args.finetuning_type != "lora":
        args.rank = 0
        args.alpha = 0

    assert args.train_datasets is not None, "Train datasets must be provided"

    epochs = {
        "Cars": 35, "DTD": 76, "EuroSAT": 12,
        "GTSRB": 11, "MNIST": 5, "RESISC45": 15,
        "SUN397": 14, "SVHN": 4, "ImageNet": 4,
        "CIFAR10": 10, "CIFAR100": 10, "STL10": 10
    }

    SEED = args.seed
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    for dataset_name in args.train_datasets:
        args.epochs = epochs[dataset_name] if args.epochs is None else args.epochs
        args.train_dataset = dataset_name
        args.result = os.path.join(
            args.result_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"finetuned_on_{dataset_name}.json"
        )
        args.fig = os.path.join(
            args.fig_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"finetuned_on_{dataset_name}.jpg"
        )

        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)

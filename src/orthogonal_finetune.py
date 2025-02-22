import copy
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
from datasets.augmentation import get_augmented_preprocess_fn
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.mixed_dataset import MixedDataset
from datasets.registry import get_dataset
from distributed import cleanup_ddp, distribute_loader, setup_ddp
from eval import evaluate
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from task_vectors import TaskVector
from utils import cosine_lr, LabelSmoothing


class OrthLoss(nn.Module):
    """Orthogonal loss function."""
    def __init__(self, adjust_type: str) -> None:
        super().__init__()
        self.adjust_type = adjust_type

    def forward(self, image_encoder: ImageEncoder) -> torch.Tensor:
        num_layers = image_encoder.model.visual.transformer.layers
        embeds = ["q", "k", "v", "out"]
        orth_loss = []
        for i in range(num_layers):
            for embed in embeds:
                U: torch.Tensor = getattr(
                    image_encoder.model.visual.transformer.resblocks[i].attn,
                    f"{embed}_proj"
                ).Delta.U
                UhU = U.T @ U
                eye = torch.eye(UhU.shape[0], device=UhU.device)
                if self.adjust_type == "fro":
                    orth_loss.append(torch.linalg.matrix_norm(UhU - eye, ord="fro"))
                elif self.adjust_type == "spec":
                    orth_loss.append(torch.linalg.matrix_norm(UhU - eye, ord=2))
                else:
                    raise ValueError(f"Invalid adjust type: {self.adjust_type}")
        return sum(orth_loss)


@torch.no_grad()
def calculate_norm(image_encoder: ImageEncoder) -> torch.Tensor:
    total_norm = 0.0
    count = 0
    state_dict: Dict[str, torch.Tensor] = image_encoder.model.state_dict()
    for name, param in state_dict.items():
        if "Delta.U" in name:
            total_norm += (param.T @ param).norm(p="fro")
            count += 1
    return total_norm / count


def orthogonal_finetune(rank: int, args: Args) -> ImageEncoder:
    """Finetune orthogonal matrices of a pretrained model on a single dataset."""
    if args.wandb:
        config_dict = vars(args)
        run = wandb.init(
            project="Task Arithmetic",
            group=f"orthogonal_finetune_{args.finetuning_type}",
            name=f"{args.model_architecture}_{args.pretrained}_{args.train_dataset}_{args.seed}",
            config=config_dict,
            settings=wandb.Settings(start_method="thread")
        )

    print("=" * 100)
    print(f"Orthogonal Finetuning {args.model_architecture} on {args.train_dataset} with {args.finetuning_type} finetuning")
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
    print(f"Loading a {args.pretrained} pretrained encoder")
    image_encoder = ImageEncoder(args, keep_lang=False)

    # Load task vector
    print(f"Loading a {args.pretrained_to_transfer} task vector")
    task_vector = TaskVector.load_vector(os.path.join(
            args.model_root,
            args.model_architecture,
            args.pretrained_to_transfer,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "task_vector",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"task_vector_for_{args.train_datasets}.pt"
    ))

    image_encoder: ImageEncoder = task_vector.apply_to(image_encoder, args.lamb)
    image_encoder.freeze_except_U()

    if args.wandb:
        wandb.watch(image_encoder, log="all", log_freq=250)

    print("\nTrainable Parameters:")
    for name, param in image_encoder.named_parameters():
        if param.requires_grad:
            print(name)

    params = [p for p in image_encoder.parameters() if p.requires_grad]
    assert len(params) > 0, "No trainable parameters found"

    optimizer = optim.AdamW(
        params=params, lr=args.lr, weight_decay=args.wd
    )

    preprocess_fn = image_encoder.train_preprocess
    preprocess_fn = get_augmented_preprocess_fn(preprocess_fn, 0.8)

    if args.ls > 0.0:
        ce_loss_fn = LabelSmoothing(args.ls)
    else:
        ce_loss_fn = nn.CrossEntropyLoss()
    orth_loss_fn = OrthLoss(args.adjust_type)

    if args.dataset_type == "cycle":
        dataset_dict = {
            dataset_name: get_dataset(
                dataset_name+"Val", preprocess_fn, args.dataset_root,
                args.batch_size, args.num_workers
            ) for dataset_name in args.train_datasets
        }
        for dataset_name in args.train_datasets:
            dataset_dict[dataset_name].train_dataset = MixedDataset(
                [dataset_name+"Val"], args.dataset_root,
                args.num_images, args.num_augments, preprocess_fn
            )
            dataset_dict[dataset_name].train_loader = torch.utils.data.DataLoader(
                dataset_dict[dataset_name].train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
        total_train_time = 0.0
        total_val_time = 0.0
        for epoch in range(args.epochs):
            train_losses = []
            train_ce_losses = []
            train_orth_losses = []
            train_accs = []
            training_batch_times = []
            val_losses = []
            val_ce_losses = []
            val_orth_losses = []
            val_accs = []
            validation_batch_times = []
            task_vector_norms = []
            learning_rates = []
            for dataset_name in args.train_datasets:
                args.train_dataset = dataset_name
                classification_head = get_classification_head(args, dataset_name)
                classifier = ImageClassifier(image_encoder, classification_head)
                classifier.freeze_head()
                classifier = classifier.to(rank)
                train_dataloader = get_dataloader(
                    dataset_dict[dataset_name],
                    is_train=True,
                    args=args,
                    image_encoder=None
                )
                val_dataloader = get_dataloader(
                    dataset_dict[dataset_name],
                    is_train=False,
                    args=args,
                    image_encoder=None
                )
                train_num_batches = len(train_dataloader)
                val_num_batches = len(val_dataloader)

                scheduler = cosine_lr(
                    optimizer, args.lr, args.warmup_length,
                    args.epochs * train_num_batches // args.grad_accum_steps
                )

                ddp_train_loader = distribute_loader(train_dataloader)
                ddp_val_loader = distribute_loader(val_dataloader)
                ddp_classifier = torch.nn.parallel.DistributedDataParallel(
                    classifier, device_ids=[rank],
                    find_unused_parameters=False, output_device=rank
                )

                epoch_train_start_time = time.time()
                ddp_classifier.train()
                ddp_train_loader.sampler.set_epoch(epoch)
                for i, batch in enumerate(ddp_train_loader):
                    batch_start_time = time.time()
                    train_step = i // args.grad_accum_steps + epoch * train_num_batches // args.grad_accum_steps
                    batch = maybe_dictionarize(batch)
                    inputs = batch["images"].to(rank)
                    labels = batch["labels"].to(rank).flatten()
                    dataset_labels = batch["metadata"]

                    logits = ddp_classifier(inputs)
                    predictions = torch.argmax(logits, dim=1)

                    train_ce_loss = ce_loss_fn(logits, labels)
                    train_orth_loss = orth_loss_fn(ddp_classifier.module.image_encoder)

                    train_loss = train_ce_loss + args.alpha * train_orth_loss
                    train_loss.backward()

                    train_corrects = (predictions == labels).sum().item()
                    train_acc = train_corrects / len(labels)

                    norm = calculate_norm(ddp_classifier.module.image_encoder)

                    train_losses.append(train_loss.item())
                    train_ce_losses.append(train_ce_loss.item())
                    train_orth_losses.append(train_orth_loss.item())
                    train_accs.append(train_acc)
                    task_vector_norms.append(norm.item())
                    learning_rates.append(optimizer.param_groups[0]["lr"])

                    if (i + 1) % args.grad_accum_steps == 0:
                        scheduler(train_step)
                        nn.utils.clip_grad_norm_(params, max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                        training_batch_time = time.time() - batch_start_time
                        training_batch_times.append(training_batch_time)
                        percent_complete = 100 * i / len(ddp_train_loader)
                        print(
                            f"Training on {args.train_dataset}\t Epoch: {epoch + 1}/{args.epochs}\t"
                            f"Batch: {i + 1}/{len(ddp_train_loader)} "
                            f"({percent_complete:.2f}%)\t"
                            f"Step: {train_step + 1}/{args.epochs * train_num_batches // args.grad_accum_steps}\t"
                            f"Training Loss: {train_loss.item():.4f}\t"
                            f"Training CE Loss: {train_ce_loss.item():.4f}\t"
                            f"Training Orth Loss: {train_orth_loss.item():.4f}\t"
                            f"Training Accuracy: {train_acc:.2%}\t"
                            f"Task Vector Norm: {norm:.4f}\t"
                            f"lr: {optimizer.param_groups[0]['lr']:.8f}\t"
                            f"Training Batch Time: {training_batch_time:.2f}s", flush=True
                        )

                    if args.save:
                        if train_step % 200 == 0:
                            filename = os.path.join(
                                model_dir,
                                f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
                                f"rank_{args.rank}_alpha_{args.alpha}",
                                "orthogonal_finetune",
                                f"bs_{args.batch_size}_seed_{args.seed}",
                                f"{args.train_datasets}",
                                f"orthogonal_finetune_on_{args.train_dataset}_step_{train_step}.pt"
                            )
                            image_encoder = copy.deepcopy(ddp_classifier.module.image_encoder).to("cpu")
                            task_vector = TaskVector(
                                pretrained_checkpoint=ImageEncoder(args, keep_lang=False),
                                finetuned_checkpoint=image_encoder
                            )
                            task_vector.save_vector(filename)

                total_train_time += time.time() - epoch_train_start_time
                print(f"\nEpoch: {epoch + 1} Training Time: {total_train_time:.2f}s\n")

                # Validation
                epoch_val_start_time = time.time()
                ddp_classifier.eval()
                ddp_val_loader.sampler.set_epoch(epoch)
                with torch.no_grad():
                    for i, batch in enumerate(ddp_val_loader):
                        batch_start_time = time.time()
                        val_step = i // args.grad_accum_steps + epoch * val_num_batches // args.grad_accum_steps
                        batch = maybe_dictionarize(batch)
                        inputs = batch["images"].to(rank)
                        labels = batch["labels"].to(rank)

                        logits = ddp_classifier(inputs)
                        predictions = torch.argmax(logits, dim=1)

                        val_ce_loss = ce_loss_fn(logits, labels)
                        val_orth_loss = orth_loss_fn(ddp_classifier.module.image_encoder)
                        val_loss = val_ce_loss + args.alpha * val_orth_loss
                        val_losses.append(val_loss.item())
                        val_ce_losses.append(val_ce_loss.item())
                        val_orth_losses.append(val_orth_loss.item())

                        val_corrects = (predictions == labels).sum().item()
                        val_acc = val_corrects / len(labels)
                        val_accs.append(val_acc)

                        if (i + 1) % args.grad_accum_steps == 0:
                            validation_batch_time = time.time() - batch_start_time
                            validation_batch_times.append(validation_batch_time)
                            percent_complete = 100 * i / len(ddp_val_loader)
                            print(
                                f"Validation on {args.train_dataset}\t Epoch: {epoch + 1}/{args.epochs}\t"
                                f"Batch: {i + 1}/{len(ddp_val_loader)} "
                                f"({percent_complete:.2f}%)\t"
                                f"Step: {val_step + 1}/{args.epochs * val_num_batches // args.grad_accum_steps}\t"
                                f"Validation Loss: {val_loss.item():.4f}\t"
                                f"Validation CE Loss: {val_ce_loss.item():.4f}\t"
                                f"Validation Orth Loss: {val_orth_loss.item():.4f}\t"
                                f"Validation Accuracy: {val_acc:.2%}\t"
                                f"Validation Batch Time: {validation_batch_time:.2f}s", flush=True
                            )

                total_val_time += time.time() - epoch_val_start_time
                print(f"\nEpoch: {epoch + 1} Validation Time: {total_val_time:.2f}s\n")

                if args.wandb:
                    run.log({
                        "Epoch": epoch + 1,
                        "Training Loss": np.mean(train_losses),
                        "Training CE Loss": np.mean(train_ce_losses),
                        "Training Orth Loss": np.mean(train_orth_losses),
                        "Training Accuracy": np.mean(train_accs),
                        "Task Vector Norm": np.mean(task_vector_norms),
                        "Learning Rate": np.mean(learning_rates),
                        "Validation Loss": np.mean(val_losses),
                        "Validation CE Loss": np.mean(val_ce_losses),
                        "Validation Orth Loss": np.mean(val_orth_losses),
                        "Validation Accuracy": np.mean(val_accs),
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

    elif args.dataset_type == "mix":
        pass
    elif args.dataset_type == "consecutive":
        pass

    # Save the finetuned model
    if args.save:
        filename = os.path.join(
            model_dir,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "orthogonal_finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"orthogonal_finetuned_image_encoder_on_{args.train_dataset}.pt"
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
            "orthogonal_finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"orthogonal_finetuned_task_vector_on_{args.train_dataset}.pt"
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

    SEED = args.seed
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    torch.multiprocessing.spawn(orthogonal_finetune, args=(args,), nprocs=args.world_size)

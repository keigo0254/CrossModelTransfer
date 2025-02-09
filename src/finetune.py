from argparse import Namespace
import copy
import os
import random
import time

import numpy as np
import torch
import torch.distributed
import wandb

from args import Args
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from distributed import cleanup_ddp, distribute_loader, setup_ddp
from eval import evaluate
from heads import get_classification_head
from linear import LinearImageEncoder
from lora import LoRAImageEncoder
from modeling import ImageClassifier, ImageEncoder
from utils import cosine_lr, LabelSmoothing


def calc_norm(encoder: ImageEncoder) -> torch.Tensor:
    norm = 0.0
    cnt = 0
    for name, param in encoder.named_parameters():
        if "D" in name:
            norm += param.norm(p="fro")
            cnt += 1
    norm /= cnt
    return norm


def finetune(rank, args: Args) -> ImageEncoder:
    if args.wandb:
        config_dict = {
            "model_architecture": args.model_architecture,
            "pretrain": args.pretrained,
            "finetuning_type": args.finetuning_type,
            "train_dataset": args.train_dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "wd": args.wd,
            "ls": args.ls,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "warmup_length": args.warmup_length,
            "seed": args.seed
        }
        if args.finetuning_type == "lora":
            config_dict["r"] = args.rank
            config_dict["alpha"] = args.alpha
        run = wandb.init(
            project="TaskArithmetic",
            group="Finetune",
            name=(f"{args.model_architecture}_{args.pretrained}"
                  f"_{args.finetuning_type}_{args.train_dataset}_{args.seed}"),
            config=config_dict
        )
    print("="*100)
    print(f"Finetuning {args.pretrained} {args.model_architecture}"
          f" on {args.train_dataset}, method: {args.finetuning_type}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*100)

    setup_ddp(rank, args.world_size, port=12355)

    assert args.train_dataset is not None, "Please provide a training dataset"
    model_dir = os.path.join(
        args.model_root,
        args.model_architecture,
        args.pretrained,
        args.finetuning_type,
    )

    if args.pretrained_model_path is not None:
        print(f"Loading model from {args.pretrained_model_path}")
        image_encoder = ImageEncoder.load(args.pretrained_model_path)
    elif args.finetuning_type == "lora":
        print(f"Loading model from {args.pretrained}")
        image_encoder = LoRAImageEncoder(args, keep_lang=False)
        filename = f"zeroshot_{args.finetuning_type}_{args.rank}.pt"
        image_encoder.save(os.path.join(model_dir, filename))
        image_encoder.freeze()
    elif args.finetuning_type == "linear":
        print(f"Loading model from {args.pretrained}")
        image_encoder = LinearImageEncoder(args, keep_lang=False)
        filename = f"zeroshot_{args.finetuning_type}.pt"
        image_encoder.save(os.path.join(model_dir, filename))
        image_encoder.freeze()
    else:
        print("Building an image encoder")
        image_encoder: ImageEncoder = ImageEncoder(args, keep_lang=False)
        filename = f"zeroshot_{args.finetuning_type}.pt"
        image_encoder.save(os.path.join(model_dir, filename))

    classification_head = get_classification_head(
        args, args.train_dataset
    )

    model = ImageClassifier(
        image_encoder, classification_head
    )
    model.freeze_head()
    model = model.to(rank)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    preprocess_fn = model.train_preprocess

    dataset = get_dataset(
        args.train_dataset,
        preprocess_fn,
        location=args.dataset_root,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset,
        is_train=True,
        args=args,
        image_encoder=None
    )
    num_batches = len(dataset.train_loader)

    ddp_loader = distribute_loader(dataloader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank
    )

    if args.ls > 0.0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    assert len(params) > 0, "No parameters to train."

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    # Gradient Accumulation
    scheduler = cosine_lr(
        optimizer, args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.grad_accum_steps
    )

    # Training Loop
    print("Training...")
    train_start_time = time.time()
    for epoch in range(args.epochs):
        ddp_model.train()
        ddp_loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(ddp_loader):
            batch_start_time = time.time()
            step = i // args.grad_accum_steps \
                    + epoch * num_batches // args.grad_accum_steps

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(rank)
            labels = batch["labels"].to(rank)

            logits = ddp_model(inputs)
            predictions = torch.argmax(logits, dim=1)

            loss = loss_fn(logits, labels)
            corrects = (predictions == labels).sum().item()
            acc = corrects / inputs.size(0)            

            loss.backward()

            norm = calc_norm(ddp_model.module.image_encoder)

            if (i + 1) % args.grad_accum_steps == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                batch_time = time.time() - batch_start_time

                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch}/{args.epochs} "
                    f"step: {step}"
                    f"[{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Train Loss: {loss.item():.6f}\t"
                    f"Train Accuracy: {acc:.6f}\t"
                    f"Norm: {norm:.6f}\t"
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}\t"
                    f"Batch Time: {batch_time:.6f}", flush=True
                )

                # wandbにログを送信
                if args.wandb:
                    run.log({
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "train_accuracy": acc,
                        "norm": norm,
                        "lr": optimizer.param_groups[0]["lr"],
                        "batch_time": batch_time
                    })
            
            if args.save:
                if step % 200 == 0:            
                    filename = os.path.join(
                        model_dir,
                        f"{args.lr}_{args.rank}_{args.alpha}"
                        if args.finetuning_type == "lora" else f"{args.lr}",
                        "finetune",
                        f"finetuned_on_{args.train_dataset}_{args.seed}_{step}.pt"
                    )
                    image_encoder_to_save = copy.deepcopy(ddp_model.module.image_encoder)
                    image_encoder_to_save.save(filename)

    train_time = time.time() - train_start_time
    print(f"Training Time: {train_time:.6f}")

    if args.save:
        filename = os.path.join(
            model_dir,
            f"{args.lr}_{args.rank}_{args.alpha}"
            if args.finetuning_type == "lora" else f"{args.lr}",
            "finetune",
            f"finetuned_on_{args.train_dataset}_{args.seed}.pt"
        )
        ddp_model.module.image_encoder.save(filename)

    evaluate(ddp_model.module.image_encoder, args)

    cleanup_ddp()
    if args.wandb:
        run.finish()

    return ddp_model.module.image_encoder


if __name__ == "__main__":
    args: Args = Args.from_args()
    assert args.batch_size % args.grad_accum_steps == 0, \
        (f"Batch size must be divisible by grad_accum_steps: "
        f"batch_size: {args.batch_size}, "
        f"grad_accum_steps: {args.grad_accum_steps}")
    args.batch_size = args.batch_size // args.grad_accum_steps

    # Number of epochs for each dataset
    epochs = {
        "Cars": 35, "DTD": 76, "EuroSAT": 12,
        "GTSRB": 11, "MNIST": 5, "RESISC45": 15,
        "SUN397": 14, "SVHN": 4, "ImageNet": 4
    }

    # Finetuning
    for dataset in args.train_datasets:
        # Set seed
        SEED = args.seed
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        args.epochs = epochs[dataset]
        args.train_dataset = dataset

        args.result = os.path.join(
            args.result_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"{args.lr}_{args.rank}_{args.alpha}"
            if args.finetuning_type == "lora" else f"{args.lr}",
            "finetune",
            f"{args.train_dataset}_{args.seed}.json"
        )
        args.fig = os.path.join(
            args.fig_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"{args.lr}_{args.rank}_{args.alpha}"
            if args.finetuning_type == "lora" else f"{args.lr}",
            "finetune",
            f"{args.train_dataset}_{args.seed}.jpg"
        )

        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)

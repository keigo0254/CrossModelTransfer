import argparse
import os
from typing import List, Sequence

from classopt import classopt, config
import torch


@classopt(default_long=True, default_short=False)
class Args:
    dataset_root: str = config(default=os.path.expanduser("dataset"), help="Path to dataset root")
    model_root: str = config(default=os.path.expanduser("model"), help="Path to model root")
    cache_root: str = config(default=os.path.expanduser(".cache"), help="Path to cache root")
    result_root: str = config(default=os.path.expanduser("result"), help="Path to result root")
    fig_root: str = config(default=os.path.expanduser("fig"), help="Path to figure root")
    result : str = config(default=None, help="Path to result file")
    fig : str = config(default=None, help="Path to figure file")

    pretrained_model_path: str = config(default=None, help="Path to pretrained model")
    base_model_path: str = config(default=None, help="Path to base model for task arithmetic")

    model_architecture: str = config(default="ViT-B-32", help="Model architecture")    # ViT-B-32, ViT-B-16, ViT-L-14
    pretrained: str = config(default="openai", help="Pretrained type")                 # openai, laion400m_e31, etc.
    finetuning_type: str = config(default="full", help="Finetuning type")              # full, linear, lora, singular
    adjust_type: str = config(default="none", help="Adjust type")                      # regularize, qr, cayley, reconstruct

    train_datasets: lambda x: x.split(",") = config(default=None, help="Train datasets")   # Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN, STL10, CIFAR10, CIFAR100, ImageNet
    eval_datasets: lambda x: x.split(",") = config(default=None, help="Evaluate datasets") # Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN, STL10, CIFAR10, CIFAR100, ImageNet

    save: bool = config(default=False, help="Whether to save the model")
    wandb: bool = config(default=False, help="Whether to use wandb")

    seed: int = config(default=42, help="Random seed")
    batch_size: int = config(default=32, help="Batch size")
    num_workers: int = config(default=4, help="Number of workers")
    epochs: int = config(default=None, help="Number of epochs")
    lr: float = config(default=1e-5, help="Learning rate")
    wd: float = config(default=0.1, help="Weight decay")
    ls: float = config(default=0.0, help="Label smoothing")
    warmup_length: int = config(default=500, help="Warmup length")
    grad_accum_steps: int = config(default=1, help="Gradient accumulation steps")
    world_size: int = config(default=1, help="World size")

    rank: int = config(default=None, help="Matrix Rank")
    alpha: float = config(default=None, help="Scaling Coefficient for LoRA")
    lamb: float = config(default=None, help="Scaling coefficient for task arithmetic")
    dropout: float = config(default=None, help="Dropout Rate")

    num_images: int = config(default=None, help="Number of images")
    num_augments: int = config(default=None, help="Number of augments")

    device: torch.device = config(default=torch.device("cuda"), help="Device")


if __name__ == "__main__":
    args: Args = Args.from_args()
    print(args)

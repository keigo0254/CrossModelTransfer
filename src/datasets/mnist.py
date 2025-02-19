import os
from typing import List

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MNIST:
    def __init__(
        self,
        preprocess: transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        self.train_dataset: datasets.MNIST = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset: datasets.MNIST = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        self.test_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


if __name__ == "__main__":
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    dataset = MNIST(preprocess, location=root)
    for i, (data, target) in enumerate(dataset.train_loader):
        print(data.shape, target)
        break

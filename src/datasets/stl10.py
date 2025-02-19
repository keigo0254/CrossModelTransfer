import os
from typing import List

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class STL10:
    def __init__(
        self,
        preprocess: transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        location = os.path.join(location, "stl10")

        self.train_dataset: datasets.STL10 = datasets.STL10(
            root=location,
            download=True,
            split="train",
            transform=preprocess
        )

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset: datasets.STL10 = datasets.STL10(
            root=location,
            download=True,
            split="test",
            transform=preprocess
        )

        self.test_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames: List[str] = self.train_dataset.classes


if __name__ == "__main__":
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    dataset = STL10(preprocess, location=root)
    for i, (data, target) in enumerate(dataset.train_loader):
        print(data.shape, target)
        break

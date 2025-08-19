import os
from typing import List

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class SUN397:
    def __init__(
        self,
        preprocess: transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        traindir = os.path.join(location, "sun397", "train")
        valdir = os.path.join(location, "sun397", "val")

        self.train_dataset: datasets.ImageFolder = datasets.ImageFolder(
            traindir,
            transform=preprocess
        )
        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset: datasets.ImageFolder = datasets.ImageFolder(
            valdir,
            transform=preprocess
        )
        self.test_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.classnames: List[str] = [
            idx_to_class[i][8:].replace("_", " ")
            for i in range(len(idx_to_class))
        ]

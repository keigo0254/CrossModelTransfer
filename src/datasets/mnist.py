import os
from typing import List

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

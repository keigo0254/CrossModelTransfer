import os

import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100


class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser("dataset"),
                 batch_size=32,
                 num_workers=4):

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = CIFAR100(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

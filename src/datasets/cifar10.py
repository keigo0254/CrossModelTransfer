import os

import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset


cifar_classnames = ["airplane", "automobile", "bird",
                    "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CIFAR10:
    def __init__(self, preprocess,
                 location=os.path.expanduser("dataset"),
                 batch_size=32,
                 num_workers=4):

        self.train_dataset = PyTorchCIFAR10(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR10(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x


class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(
            root=None, transform=transform, target_transform=target_transform)
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = CIFAR10(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

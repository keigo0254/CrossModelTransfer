import os
import re

import torch
import torchvision
import torchvision.datasets as datasets


def pretify_classname(classname: str) -> str:
    words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
    words = [i.lower() for i in words]
    out = " ".join(words)
    if out.endswith("al"):
        return out + " area"
    return out


class EuroSATBase:
    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        test_split: str,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        # Setup dataset paths
        traindir = os.path.join(location, "EuroSAT_splits", "train")
        testdir = os.path.join(location, "EuroSAT_splits", test_split)

        # Setup training data
        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Setup test data
        self.test_dataset = datasets.ImageFolder(
            testdir, transform=preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # Setup and convert class names
        idx_to_class = dict(
            (v, k) for k, v in self.train_dataset.class_to_idx.items()
        )
        self.classnames = [
            idx_to_class[i].replace("_", " ")
            for i in range(len(idx_to_class))
        ]
        self.classnames = [
            pretify_classname(c) for c in self.classnames
        ]

        # Convert to OpenAI format class names
        ours_to_open_ai = {
            "annual crop": "annual crop land",
            "forest": "forest",
            "herbaceous vegetation": "brushland or shrubland",
            "highway": "highway or road",
            "industrial area": "industrial buildings or commercial buildings",
            "pasture": "pasture land",
            "permanent crop": "permanent crop land",
            "residential area": "residential buildings or homes or apartments",
            "river": "river",
            "sea lake": "lake or sea",
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        super().__init__(
            preprocess, "test",
            location, batch_size, num_workers
        )


class EuroSATVal(EuroSATBase):
    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        super().__init__(
            preprocess, "val",
            location, batch_size, num_workers
        )


if __name__ == "__main__":
    # Test functionality
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = EuroSAT(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

    d = EuroSATVal(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

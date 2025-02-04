import os

import torch
import torchvision.datasets as datasets


class STL10:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser("dataset"),
                 batch_size=32,
                 num_workers=12):

        location = os.path.join(location, "stl10")
        self.train_dataset = datasets.STL10(
            root=location,
            download=True,
            split="train",
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.STL10(
            root=location,
            download=True,
            split="test",
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = self.train_dataset.classes


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = STL10(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

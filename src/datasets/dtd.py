import os

import torch
import torchvision.datasets as datasets


class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser("dataset"),
                 batch_size=32,
                 num_workers=4):
        # Data loading code
        traindir = os.path.join(location, "dtd", "train")
        valdir = os.path.join(location, "dtd", "val")

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            "_", " ") for i in range(len(idx_to_class))]


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = DTD(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

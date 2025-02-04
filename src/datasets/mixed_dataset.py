import random
from typing import Dict, List

from PIL import ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms


class MixedDataset(Dataset):
    def __init__(self, dataset_list: List[str], root: str, num_images: int, preprocess):
        super().__init__()
        self.dataset_list = dataset_list
        self.root = root
        self.dataset = []
        self.preprocess = preprocess

        for dataset_name in dataset_list:
            dataset = get_dataset(dataset_name, transforms.ToTensor(), root, batch_size=1)
            dataloader = get_dataloader(dataset, is_train=True, args=None)
            for i, (img, label) in enumerate(dataloader):
                if i >= num_images:
                    break
                self.dataset.append((img, label, dataset_name))

        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, dataset_name = self.dataset[idx]

        if self.preprocess is not None:
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)
                img = transforms.ToPILImage()(img)
            img = self.preprocess(img)

        return img, label, dataset_name


if __name__ == "__main__":
    import os

    import open_clip

    from augmentation import get_augmented_preprocess_fn
    from common import get_dataloader
    from registry import get_dataset


    root = os.path.expanduser("dataset")

    _, preprocess_fn, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )
    preprocess_fn = get_augmented_preprocess_fn(preprocess_fn)

    dataset_list = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN"
    ]
    num_images = 100

    mixed_dataset = MixedDataset(dataset_list, root, num_images, preprocess_fn)
    mixed_dataloader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)

    for i, (img, label, dataset_name) in enumerate(mixed_dataloader):
        print(i, img.shape, label, dataset_name)
        break    

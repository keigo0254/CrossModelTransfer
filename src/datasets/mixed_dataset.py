# 複数データセットを混ぜたデータセットを作成する
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .common import get_dataloader, maybe_dictionarize
from .registry import get_dataset


class MixedDataset(Dataset):
    def __init__(self, dataset_list: List[str], root: str, num_images: int, preprocess):
        super().__init__()
        self.dataset_list = dataset_list
        self.root = root
        self.dataset = []
        self.preprocess = preprocess

        print(f"Creating mixed dataset with {num_images} images from each dataset")
        for dataset_name in dataset_list:
            print(f"Loading {num_images} images from {dataset_name}")
            dataset = get_dataset(dataset_name, transforms.ToTensor(), root, batch_size=1, num_workers=0)
            dataloader = get_dataloader(dataset, is_train=True, args=None)
            for i, batch in enumerate(dataloader):
                if i >= num_images:
                    break
                batch = maybe_dictionarize(batch)
                self.dataset.append((batch["images"], batch["labels"], dataset_name))

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

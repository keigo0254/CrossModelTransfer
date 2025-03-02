import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .common import get_dataloader, maybe_dictionarize
from .registry import get_dataset


class MixedDataset(Dataset):
    """Dataset class that combines multiple datasets.

    Take samples from multiple source datasets and combine them into a single
    mixed dataset. The samples are randomly shuffled.
    """
    def __init__(
        self,
        dataset_list: List[str],
        root: str,
        num_images: int,
        num_augments: int,
        preprocess: transforms.Compose
    ) -> None:
        super().__init__()
        self.dataset_list = dataset_list
        self.root = root
        self.dataset = []
        self.preprocess = preprocess
        self.num_images = num_images
        self.num_augments = num_augments

        print(f"Creating mixed dataset with {num_images} images from each dataset")
        for dataset_name in dataset_list:
            print(f"Loading {num_images} images from {dataset_name}")
            # Load each dataset
            dataset = get_dataset(
                dataset_name,
                transforms.ToTensor(),
                root,
                batch_size=1,
                num_workers=0
            )
            dataloader = get_dataloader(dataset, is_train=True, args=None)

            # Extract specified number of samples
            for i, batch in enumerate(dataloader):
                if i >= self.num_images:
                    break
                batch = maybe_dictionarize(batch)
                self.dataset.append((batch["images"], batch["labels"].squeeze(0), dataset_name))

        # Randomly shuffle samples
        random.shuffle(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset) * self.num_augments

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img, label, dataset_name = self.dataset[idx // self.num_augments]

        # Apply preprocessing if specified
        if self.preprocess is not None:
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)
                img = transforms.ToPILImage()(img)
            img = self.preprocess(img)

        return img, label, dataset_name

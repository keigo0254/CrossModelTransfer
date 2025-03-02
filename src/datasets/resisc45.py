import abc
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader
from torchvision.transforms import Compose


class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self) -> str:
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        image, label = self._load_image(index)

        if self.transforms is not None:
            return self.transforms(image), label

        return image, label

    def __len__(self) -> int:
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        img, label = ImageFolder.__getitem__(self, index)
        label = torch.tensor(label)
        return img, label


class RESISC45Dataset(VisionClassificationDataset):
    directory = "resisc45/NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }
    classes = [
        "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
        "bridge", "chaparral", "church", "circular_farmland", "cloud",
        "commercial_area", "dense_residential", "desert", "forest", "freeway",
        "golf_course", "ground_track_field", "harbor", "industrial_area",
        "intersection", "island", "lake", "meadow", "medium_residential",
        "mobile_home_park", "mountain", "overpass", "palace", "parking_lot",
        "railway", "railway_station", "rectangular_farmland", "river",
        "roundabout", "runway", "sea_ice", "ship", "snowberg",
        "sparse_residential", "stadium", "storage_tank", "tennis_court",
        "terrace", "thermal_power_station", "wetland",
    ]

    def __init__(
        self,
        root: str = os.path.expanduser("dataset"),
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        assert split in self.splits
        self.root = root

        valid_fns = set()
        with open(os.path.join(self.root, "resisc45", f"resisc45-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )


class RESISC45:
    def __init__(
        self,
        preprocess: Compose,
        location: str = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        self.train_dataset = RESISC45Dataset(
            root=location,
            split="train",
            transforms=preprocess
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = RESISC45Dataset(
            root=location,
            split="test",
            transforms=preprocess
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = [" ".join(c.split("_")) for c in RESISC45Dataset.classes]

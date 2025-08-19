import os
import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image
import torch
import torchvision
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    verify_str_arg
)
from torchvision.datasets.vision import VisionDataset


class PytorchStanfordCars(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError(
                "Scipy is not found. "
                "This dataset needs to have scipy installed: pip install scipy"
            )

        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = (
                self._base_folder / "cars_test_annos_withlabels.mat"
            )
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,
            )
            for annotation in sio.loadmat(
                self._annotations_mat_path,
                squeeze_me=True
            )["annotations"]
        ]

        self.classes = sio.loadmat(
            str(devkit / "cars_meta.mat"),
            squeeze_me=True
        )["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            print(f"check_exists: {self._check_exists()}")
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )

        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url=(
                    "https://ai.stanford.edu/~jkrause/car196/"
                    "cars_test_annos_withlabels.mat"
                ),
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return (
            self._annotations_mat_path.exists()
            and self._images_base_path.is_dir()
        )


class Cars:
    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        self.train_dataset = PytorchStanfordCars(
            location,
            "train",
            preprocess,
            download=False
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = PytorchStanfordCars(
            location,
            "test",
            preprocess,
            download=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        idx_to_class = dict(
            (v, k) for k, v in self.train_dataset.class_to_idx.items()
        )
        self.classnames = [
            idx_to_class[i].replace("_", " ")
            for i in range(len(idx_to_class))
        ]

# CIFAR-10のデータセットを扱うためのクラスを定義
import os
from typing import Any, List, Tuple

from PIL.Image import Image
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset


cifar_classnames = ["airplane", "automobile", "bird",
                    "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CIFAR10:
    """
    CIFAR-10データセットのラッパークラス

    Attributes:
        train_dataset (PyTorchCIFAR10): 学習用データセット
        train_loader (torch.utils.data.DataLoader): 学習用データローダー
        test_dataset (PyTorchCIFAR10): テスト用データセット
        test_loader (torch.utils.data.DataLoader): テスト用データローダー
        classnames (List[str]): クラス名のリスト
    """
    def __init__(self, preprocess: torchvision.transforms.Compose,
                 location: str | os.PathLike = os.path.expanduser("dataset"),
                 batch_size: int = 32,
                 num_workers: int = 4) -> None:
        """
        CIFAR-10データセットを扱うクラスを初期化

        Args:
            preprocess (torchvision.transforms.Compose): 前処理関数
            location (str | os.PathLike, optional): データセットの保存先ディレクトリ. \
                Defaults to os.path.expanduser("dataset").
            batch_size (int, optional): バッチサイズ. Defaults to 32.
            num_workers (int, optional): データローダーの並列数. Defaults to 4.
        """
        self.train_dataset = PyTorchCIFAR10(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR10(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


def convert(x: np.ndarray | Any) -> Image | Any:
    """
    numpy.ndarrayをPIL.Imageに変換する

    Args:
        x (np.ndarray | Any): 変換するデータ

    Returns:
        Image | Any: 変換後のデータ
    """
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x


class BasicVisionDataset(VisionDataset):
    """
    画像分類のためのデータセットクラス(未使用)

    Attributes:
        images (List[np.ndarray | Any]): 画像データのリスト
        targets (List[int]): ラベルのリスト
    """
    def __init__(
        self,
        images: List[np.ndarray | Any],
        targets: List[int],
        transform: torchvision.transforms.Compose = None,
        target_transform: torchvision.transforms.Compose = None
    ) -> None:
        """
        画像分類のためのデータセットクラスを初期化

        Args:
            images (List[np.ndarray  |  Any]): 画像データのリスト
            targets (List[int]): ラベルのリスト
            transform (torchvision.transforms.Compose, optional): 画像の変換関数. \
                Defaults to None.
            target_transform (torchvision.transforms.Compose, optional): \
                ラベルの変換関数. Defaults to None.
        """
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(
            root=None, transform=transform, target_transform=target_transform)
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        指定したインデックスのデータを取得

        Args:
            index (int): インデックス

        Returns:
            Tuple[torch.Tensor, int]: 画像データとラベル
        """
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self) -> int:
        """
        データセットのサイズを返す

        Returns:
            int: データセットのサイズ
        """
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

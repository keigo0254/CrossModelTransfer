"""
CIFAR-10データセットを扱うモジュール

CIFAR-10データセットを読み込み、前処理を行うためのクラスを提供する。

Classes:
    CIFAR10: CIFAR-10データセットのラッパークラス
    BasicVisionDataset: 画像分類のためのデータセットクラス
"""

import os
from typing import Any, List, Tuple

from PIL.Image import Image
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset


# CIFAR-10のクラス名リスト
cifar_classnames = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class CIFAR10:
    """CIFAR-10データセットのラッパークラス

    Attributes:
        train_dataset: 学習用データセット
        train_loader: 学習用データローダー
        test_dataset: テスト用データセット
        test_loader: テスト用データローダー
        classnames: クラス名のリスト
    """

    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        """CIFAR-10データセットを扱うクラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # 訓練データの設定
        self.train_dataset = PyTorchCIFAR10(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # テストデータの設定
        self.test_dataset = PyTorchCIFAR10(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


def convert(x: np.ndarray | Any) -> Image | Any:
    """numpy.ndarrayをPIL.Imageに変換する

    Args:
        x: 変換するデータ

    Returns:
        変換後のデータ
    """
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x


class BasicVisionDataset(VisionDataset):
    """画像分類のためのデータセットクラス(未使用)

    Attributes:
        images: 画像データのリスト
        targets: ラベルのリスト
    """

    def __init__(
        self,
        images: List[np.ndarray | Any],
        targets: List[int],
        transform: torchvision.transforms.Compose = None,
        target_transform: torchvision.transforms.Compose = None
    ) -> None:
        """画像分類のためのデータセットクラスを初期化

        Args:
            images: 画像データのリスト
            targets: ラベルのリスト
            transform: 画像の変換関数. Defaults to None.
            target_transform: ラベルの変換関数. Defaults to None.
        """
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(
            root=None,
            transform=transform,
            target_transform=target_transform
        )
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """指定したインデックスのデータを取得

        Args:
            index: インデックス

        Returns:
            画像データとラベルのタプル
        """
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self) -> int:
        """データセットのサイズを返す

        Returns:
            データセットのサイズ
        """
        return len(self.targets)


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = CIFAR10(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

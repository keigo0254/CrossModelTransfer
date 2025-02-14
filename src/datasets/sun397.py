"""SUN397データセットを扱うモジュール

SUN397データセットを読み込み、前処理を行うためのクラスを提供する。

Classes:
    SUN397: SUN397データセットのラッパークラス
"""

import os
from typing import List

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class SUN397:
    """SUN397データセットのラッパークラス

    Attributes:
        train_dataset: 学習用データセット
        train_loader: 学習用データローダー
        test_dataset: テスト用データセット
        test_loader: テスト用データローダー
        classnames: クラス名のリスト
    """

    def __init__(
        self,
        preprocess: transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        """SUN397データセットのラッパークラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットのルートディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # データセットのパスを設定
        traindir = os.path.join(location, "sun397", "train")
        valdir = os.path.join(location, "sun397", "val")

        # 学習用データセットの設定
        self.train_dataset: datasets.ImageFolder = datasets.ImageFolder(
            traindir,
            transform=preprocess
        )
        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # テスト用データセットの設定
        self.test_dataset: datasets.ImageFolder = datasets.ImageFolder(
            valdir,
            transform=preprocess
        )
        self.test_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # クラス名の設定
        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.classnames: List[str] = [
            idx_to_class[i][2:].replace("_", " ")
            for i in range(len(idx_to_class))
        ]


if __name__ == "__main__":
    # 動作検証用コード
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    dataset = SUN397(preprocess, location=root)
    for i, (data, target) in enumerate(dataset.train_loader):
        print(data.shape, target)
        break

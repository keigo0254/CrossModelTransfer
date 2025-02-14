"""SVHNデータセットを扱うモジュール

SVHNデータセットを読み込み、前処理を行うためのクラスを提供する。

Classes:
    SVHN: SVHNデータセットのラッパークラス
"""

import os
from typing import List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN as PyTorchSVHN


class SVHN:
    """SVHNデータセットのラッパークラス

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
        """SVHNデータセットのラッパークラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットのルートディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # データセットのパスを設定
        modified_location = os.path.join(location, "svhn")

        # 学習用データセットの設定
        self.train_dataset: PyTorchSVHN = PyTorchSVHN(
            root=modified_location,
            download=True,
            split="train",
            transform=preprocess
        )

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # テスト用データセットの設定
        self.test_dataset: PyTorchSVHN = PyTorchSVHN(
            root=modified_location,
            download=True,
            split="test",
            transform=preprocess
        )

        self.test_loader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # クラス名の設定（0-9の数字）
        self.classnames: List[str] = [str(i) for i in range(10)]


if __name__ == "__main__":
    # 動作検証用コード
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    dataset = SVHN(preprocess, location=root)
    for i, (data, target) in enumerate(dataset.train_loader):
        print(data.shape, target)
        break

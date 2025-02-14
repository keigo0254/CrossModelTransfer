"""
CIFAR-100データセットを扱うモジュール

CIFAR-100データセットを読み込み、前処理を行うためのクラスを提供する。

Classes:
    CIFAR100: CIFAR-100データセットのラッパークラス
"""

import os

import torch
import torchvision
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100


class CIFAR100:
    """CIFAR-100データセットのラッパークラス

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
        """CIFAR-100データセットを扱うクラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # 訓練データの設定
        self.train_dataset = PyTorchCIFAR100(
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
        self.test_dataset = PyTorchCIFAR100(
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

        # クラス名の設定
        self.classnames = self.test_dataset.classes


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = CIFAR100(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

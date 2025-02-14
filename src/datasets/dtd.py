"""
DTDデータセットを扱うモジュール

Describable Textures Dataset (DTD)を読み込み、前処理を行うためのクラスを提供する。

Classes:
    DTD: DTDデータセットのラッパークラス
"""

import os

import torch
import torchvision
import torchvision.datasets as datasets


class DTD:
    """DTDデータセットのラッパークラス

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
        """DTDデータセットを扱うクラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # データセットのパスを設定
        traindir = os.path.join(location, "dtd", "train")
        valdir = os.path.join(location, "dtd", "val")

        # 訓練データの設定
        self.train_dataset = datasets.ImageFolder(
            traindir,
            transform=preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # テストデータの設定
        self.test_dataset = datasets.ImageFolder(
            valdir,
            transform=preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # クラス名の設定
        idx_to_class = {
            v: k for k, v in self.train_dataset.class_to_idx.items()
        }
        self.classnames = [
            idx_to_class[i].replace("_", " ")
            for i in range(len(idx_to_class))
        ]


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = DTD(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

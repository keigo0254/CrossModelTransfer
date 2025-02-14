"""EuroSATデータセットを扱うモジュール

EuroSATデータセットを読み込み、前処理を行うためのクラスを提供する。

Classes:
    EuroSATBase: EuroSATデータセットの基底クラス
    EuroSAT: EuroSATデータセットのラッパークラス
    EuroSATVal: EuroSATデータセットの検証用ラッパークラス
"""

import os
import re

import torch
import torchvision
import torchvision.datasets as datasets


def pretify_classname(classname: str) -> str:
    """クラス名を見やすい形式に変換する

    Args:
        classname: 変換前のクラス名

    Returns:
        変換後のクラス名
    """
    words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
    words = [i.lower() for i in words]
    out = " ".join(words)
    if out.endswith("al"):
        return out + " area"
    return out


class EuroSATBase:
    """EuroSATデータセットの基底クラス

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
        test_split: str,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        """EuroSATデータセットの基底クラスを初期化

        Args:
            preprocess: 前処理関数
            test_split: テストデータの分割名
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # データセットのパスを設定
        traindir = os.path.join(location, "EuroSAT_splits", "train")
        testdir = os.path.join(location, "EuroSAT_splits", test_split)

        # 訓練データの設定
        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # テストデータの設定
        self.test_dataset = datasets.ImageFolder(
            testdir, transform=preprocess
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # クラス名の設定と変換
        idx_to_class = dict(
            (v, k) for k, v in self.train_dataset.class_to_idx.items()
        )
        self.classnames = [
            idx_to_class[i].replace("_", " ")
            for i in range(len(idx_to_class))
        ]
        self.classnames = [
            pretify_classname(c) for c in self.classnames
        ]

        # OpenAI形式のクラス名に変換
        ours_to_open_ai = {
            "annual crop": "annual crop land",
            "forest": "forest",
            "herbaceous vegetation": "brushland or shrubland",
            "highway": "highway or road",
            "industrial area": "industrial buildings or commercial buildings",
            "pasture": "pasture land",
            "permanent crop": "permanent crop land",
            "residential area": "residential buildings or homes or apartments",
            "river": "river",
            "sea lake": "lake or sea",
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    """EuroSATデータセットのラッパークラス

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
        """EuroSATデータセットのラッパークラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        super().__init__(
            preprocess, "test",
            location, batch_size, num_workers
        )


class EuroSATVal(EuroSATBase):
    """EuroSATデータセットの検証用ラッパークラス

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
        """EuroSATデータセットの検証用ラッパークラスを初期化

        Args:
            preprocess: 前処理関数
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        super().__init__(
            preprocess, "val",
            location, batch_size, num_workers
        )


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = EuroSAT(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

    d = EuroSATVal(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

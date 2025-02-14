"""RESISC45データセットを扱うモジュール

RESISC45データセットを読み込み、前処理を行うためのクラスを提供する。

Classes:
    VisionDataset: 基本的な画像データセットの抽象基底クラス
    VisionClassificationDataset: 画像分類データセットの抽象基底クラス
    RESISC45Dataset: RESISC45データセットのラッパークラス
    RESISC45: RESISC45データセットのメインクラス
"""

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
    """基本的な画像データセットの抽象基底クラス

    地理空間情報を持たないデータセット向けの基底クラス。
    事前に定義された画像チップを持つデータセット用に設計。
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """データセット内の指定インデックスのアイテムを返す

        Args:
            index: 取得するアイテムのインデックス

        Returns:
            指定インデックスのデータとラベル

        Raises:
            IndexError: インデックスが範囲外の場合
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """データセットの長さを返す

        Returns:
            データセットの長さ
        """

    def __str__(self) -> str:
        """オブジェクトの非公式な文字列表現を返す

        Returns:
            非公式な文字列表現
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):
    """画像分類データセットの抽象基底クラス

    地理空間情報を持たない分類データセット向けの基底クラス。
    クラスごとに別フォルダに分けられた事前定義の画像チップを持つデータセット用。
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """VisionClassificationDatasetを初期化

        Args:
            root: データセットのルートディレクトリ
            transforms: 入力サンプルとターゲットを変換する関数/変換
            loader: 画像ファイルパスを受け取りPIL Imageまたはnumpy arrayを返す関数
            is_valid_file: 画像ファイルパスを受け取り有効なファイルか判定する関数
        """
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """データセット内の指定インデックスのアイテムを返す

        Args:
            index: 取得するアイテムのインデックス

        Returns:
            指定インデックスのデータとラベル
        """
        image, label = self._load_image(index)

        if self.transforms is not None:
            return self.transforms(image), label

        return image, label

    def __len__(self) -> int:
        """データセットの画像数を返す

        Returns:
            データセットの長さ
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """1つの画像とそのクラスラベルを読み込む

        Args:
            index: 取得するアイテムのインデックス

        Returns:
            画像とそのクラスラベル
        """
        img, label = ImageFolder.__getitem__(self, index)
        label = torch.tensor(label)
        return img, label


class RESISC45Dataset(VisionClassificationDataset):
    """RESISC45データセット

    リモートセンシング画像シーン分類用のデータセット。
    * 31,500枚の画像(解像度0.2-30m/pixel、256x256px)
    * RGBの3チャンネル
    * 45のシーンクラス、各クラス700枚
    * 100カ国以上のGoogle Earth画像
    * 解像度、天候、照明等の高い変動性
    """

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
        """RESISC45Datasetを初期化

        Args:
            root: データセットのルートディレクトリ
            split: "train"、"val"、"test"のいずれか
            transforms: 入力サンプルとターゲットを変換する関数/変換
        """
        assert split in self.splits
        self.root = root

        # 指定されたsplitに含まれるファイル名のセットを作成
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
    """RESISC45データセットのメインクラス

    Attributes:
        train_dataset: 学習用データセット
        train_loader: 学習用データローダー
        test_dataset: テスト用データセット
        test_loader: テスト用データローダー
        classnames: クラス名のリスト
    """

    def __init__(
        self,
        preprocess: Compose,
        location: str = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        """RESISC45を初期化

        Args:
            preprocess: 前処理関数
            location: データセットのルートディレクトリ
            batch_size: バッチサイズ
            num_workers: データローダーの並列数
        """
        # 学習用データセットとローダーの設定
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

        # テスト用データセットとローダーの設定
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

        # ゼロショット学習用にクラス名のアンダースコアをスペースに置換
        self.classnames = [" ".join(c.split("_")) for c in RESISC45Dataset.classes]


if __name__ == "__main__":
    # 動作検証用コード
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        "openai",
        cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    dataset = RESISC45(preprocess, location=root)
    for i, (data, target) in enumerate(dataset.train_loader):
        print(data.shape, target)
        break

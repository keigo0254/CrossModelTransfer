"""GTSRBデータセットを扱うモジュール

German Traffic Sign Recognition Benchmark (GTSRB)データセットを読み込み、
前処理を行うためのクラスを提供する。

Classes:
    PyTorchGTSRB: GTSRBデータセットの基底クラス
    GTSRB: GTSRBデータセットのラッパークラス
"""

import csv
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import PIL
import torch
import torchvision
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg
)
from torchvision.datasets.vision import VisionDataset


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """ディレクトリ内のクラスを見つける

    Args:
        directory: ディレクトリパス

    Raises:
        FileNotFoundError: クラスフォルダが見つからない場合

    Returns:
        クラス名のリストとクラス名をインデックスに変換する辞書のタプル
    """
    classes = sorted(
        entry.name
        for entry in os.scandir(directory) if entry.is_dir()
    )
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}."
        )

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class PyTorchGTSRB(VisionDataset):
    """GTSRBデータセットの基底クラス

    Attributes:
        _split: 'train'または'test'
        _base_folder: データセットの保存先ディレクトリ
        _target_folder: データセットの保存先ディレクトリ
        _samples: 画像ファイルパスとラベルのリスト
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """GTSRBデータセットを初期化

        Args:
            root: データセットの保存先ディレクトリ
            split: 'train'または'test'. Defaults to "train".
            transform: 画像の変換関数. Defaults to None.
            target_transform: ターゲットの変換関数. Defaults to None.
            download: ダウンロードするかどうか. Defaults to False.

        Raises:
            RuntimeError: データセットが見つからない場合
        """
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / (
                "Training" if self._split == "train" else "Final_Test/Images"
            )
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        # 訓練データとテストデータでサンプルの読み込み方法が異なる
        if self._split == "train":
            _, class_to_idx = find_classes(str(self._target_folder))
            samples = make_dataset(
                str(self._target_folder),
                extensions=(".ppm",),
                class_to_idx=class_to_idx
            )
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (
                        str(self._target_folder / row["Filename"]),
                        int(row["ClassId"])
                    ) for row in csv.DictReader(
                        csv_file, delimiter=";", skipinitialspace=True
                    )
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """データセットのサイズを返す

        Returns:
            データセットのサイズ
        """
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """指定したインデックスのデータを取得

        Args:
            index: インデックス

        Returns:
            画像とターゲットのタプル
        """
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_exists(self) -> bool:
        """データセットが存在するかどうかを返す

        Returns:
            データセットが存在するかどうか
        """
        return self._target_folder.is_dir()

    def download(self) -> None:
        """データセットをダウンロードする"""
        if self._check_exists():
            return

        base_url = (
            "https://sid.erda.dk/public/archives/"
            "daaeac0d7ce1152aea9b61d9f1e19370/"
        )

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )


class GTSRB:
    """GTSRBデータセットのラッパークラス

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
        """GTSRBデータセットを初期化

        Args:
            preprocess: 前処理関数
            location: データセットの保存先ディレクトリ.
                Defaults to os.path.expanduser("dataset").
            batch_size: バッチサイズ. Defaults to 32.
            num_workers: データローダーの並列数. Defaults to 4.
        """
        # 訓練データの設定
        self.train_dataset = PyTorchGTSRB(
            root=location,
            download=True,
            split="train",
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        # テストデータの設定
        self.test_dataset = PyTorchGTSRB(
            root=location,
            download=True,
            split="test",
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # OpenAI形式のクラス名を設定
        # https://github.com/openai/CLIP/blob/main/data/prompts.md
        self.classnames = [
            "red and white circle 20 kph speed limit",
            "red and white circle 30 kph speed limit",
            "red and white circle 50 kph speed limit",
            "red and white circle 60 kph speed limit",
            "red and white circle 70 kph speed limit",
            "red and white circle 80 kph speed limit",
            "end / de-restriction of 80 kph speed limit",
            "red and white circle 100 kph speed limit",
            "red and white circle 120 kph speed limit",
            "red and white circle red car and black car no passing",
            "red and white circle red truck and black car no passing",
            "red and white triangle road intersection warning",
            "white and yellow diamond priority road",
            "red and white upside down triangle yield right-of-way",
            "stop",
            "empty red and white circle",
            "red and white circle no truck entry",
            "red circle with white horizonal stripe no entry",
            "red and white triangle with exclamation mark warning",
            "red and white triangle with black left curve approaching warning",
            "red and white triangle with black right curve approaching warning",
            "red and white triangle with black double curve approaching warning",
            "red and white triangle rough / bumpy road warning",
            "red and white triangle car skidding / slipping warning",
            "red and white triangle with merging / narrow lanes warning",
            "red and white triangle with person digging / construction / "
            "road work warning",
            "red and white triangle with traffic light approaching warning",
            "red and white triangle with person walking warning",
            "red and white triangle with child and person walking warning",
            "red and white triangle with bicyle warning",
            "red and white triangle with snowflake / ice warning",
            "red and white triangle with deer warning",
            "white circle with gray strike bar no speed limit",
            "blue circle with white right turn arrow mandatory",
            "blue circle with white left turn arrow mandatory",
            "blue circle with white forward arrow mandatory",
            "blue circle with white forward or right turn arrow mandatory",
            "blue circle with white forward or left turn arrow mandatory",
            "blue circle with white keep right arrow mandatory",
            "blue circle with white keep left arrow mandatory",
            "blue circle with white arrows indicating a traffic circle",
            "white circle with gray strike bar indicating no passing for cars "
            "has ended",
            "white circle with gray strike bar indicating no passing for trucks "
            "has ended",
        ]


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = GTSRB(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

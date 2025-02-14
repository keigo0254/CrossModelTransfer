"""データセットの登録と管理を行うモジュール

データセットクラスの登録、取得、および検証用データセットの分割を行う機能を提供する。

Classes:
    GenericDataset: 基本的なデータセットクラス

Functions:
    split_train_into_train_val: 学習データを学習用と検証用に分割
    get_dataset: データセット名から対応するデータセットインスタンスを取得
"""

import copy
import inspect
import sys
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .cars import Cars
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .dtd import DTD
from .eurosat import EuroSAT, EuroSATVal
from .gtsrb import GTSRB
from .imagenet import ImageNet
from .mnist import MNIST
from .resisc45 import RESISC45
from .stl10 import STL10
from .svhn import SVHN
from .sun397 import SUN397


# モジュール内のすべてのクラスを登録
registry: Dict[str, Type[Any]] = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset:
    """基本的なデータセットクラス

    Attributes:
        train_dataset: 学習用データセット
        train_loader: 学習用データローダー
        test_dataset: テスト用データセット
        test_loader: テスト用データローダー
        classnames: クラス名のリスト
    """

    def __init__(self) -> None:
        self.train_dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_dataset: Optional[Dataset] = None
        self.test_loader: Optional[DataLoader] = None
        self.classnames: Optional[list] = None


def split_train_into_train_val(
    dataset: GenericDataset,
    new_dataset_class_name: str,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    max_val_samples: Optional[int] = None,
    seed: int = 0
) -> GenericDataset:
    """学習データセットを学習用と検証用に分割する

    Args:
        dataset: 分割元のデータセット
        new_dataset_class_name: 新しく作成するデータセットのクラス名
        batch_size: バッチサイズ
        num_workers: データローダーの並列数
        val_fraction: 検証用データの割合 (0.0 < val_fraction < 1.0)
        max_val_samples: 検証用データの最大サンプル数. Defaults to None.
        seed: 乱数シード. Defaults to 0.

    Returns:
        分割された新しいデータセット

    Raises:
        AssertionError: val_fractionが不正な値の場合
    """
    assert 0.0 < val_fraction < 1.0, "val_fractionは0から1の間である必要があります"

    # データセットサイズの計算
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0 and train_size > 0, "分割後のデータセットサイズが0になっています"

    # データセットの分割
    trainset, valset = random_split(
        dataset.train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # MNISTValの場合の検証
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    # 新しいデータセットクラスの作成と設定
    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    # 学習用データローダーの設定
    new_dataset.train_dataset = trainset
    new_dataset.train_loader = DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 検証用データローダーの設定
    new_dataset.test_dataset = valset
    new_dataset.test_loader = DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(
    dataset_name: str,
    preprocess: Any,
    location: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    max_val_samples: int = 5000
) -> GenericDataset:
    """データセット名から対応するデータセットインスタンスを取得する

    Args:
        dataset_name: データセット名
        preprocess: 前処理関数
        location: データセットの保存場所
        batch_size: バッチサイズ. Defaults to 32.
        num_workers: データローダーの並列数. Defaults to 4.
        val_fraction: 検証用データの割合. Defaults to 0.1.
        max_val_samples: 検証用データの最大サンプル数. Defaults to 5000.

    Returns:
        データセットインスタンス

    Raises:
        AssertionError: 未対応のデータセット名が指定された場合
    """
    if dataset_name.endswith("Val"):
        # 検証用データセットの処理
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            # 基本データセットから検証用データセットを作成
            base_dataset_name = dataset_name.split("Val")[0]
            base_dataset = get_dataset(
                base_dataset_name,
                preprocess,
                location,
                batch_size,
                num_workers
            )
            dataset = split_train_into_train_val(
                base_dataset,
                dataset_name,
                batch_size,
                num_workers,
                val_fraction,
                max_val_samples
            )
            return dataset
    else:
        assert dataset_name in registry, (
            f"未対応のデータセット: {dataset_name}. "
            f"対応データセット: {list(registry.keys())}"
        )
        dataset_class = registry[dataset_name]

    # データセットインスタンスの作成
    dataset = dataset_class(
        preprocess,
        location=location,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return dataset

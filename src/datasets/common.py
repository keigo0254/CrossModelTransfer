"""データセットの読み込みやデータローダーの作成を行うモジュール

データセットの読み込み、データローダーの作成、特徴量の抽出などの
共通的な機能を提供する。

Classes:
    SubsetSampler: データセットの一部をサンプリングするクラス
    ImageFolderWithPaths: 画像フォルダのパスを保持するデータセットクラス 
    FeatureDataset: 特徴量を保持するデータセットクラス

Functions:
    maybe_dictionarize: バッチを辞書形式に変換
    get_features_helper: データローダーから特徴量を抽出
    get_features: キャッシュを考慮して特徴量を取得
    get_dataloader: データローダーを取得
"""

from argparse import Namespace
import collections
import glob
import os
import random
from typing import Any, Dict, Generator, List

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


class SubsetSampler(Sampler):
    """データセットの一部をサンプリングするためのクラス

    Attributes:
        indices (List[int]): サンプリングするインデックスのリスト
    """

    def __init__(self, indices: List[int]) -> None:
        """SubsetSamplerを初期化

        Args:
            indices: サンプリングするインデックスのリスト
        """
        self.indices = indices

    def __iter__(self) -> Generator[int, None, None]:
        """インデックスをイテレートする

        Returns:
            インデックスのジェネレータ
        """
        return (i for i in self.indices)

    def __len__(self) -> int:
        """インデックスの数を取得

        Returns:
            インデックスの数
        """
        return len(self.indices)


class ImageFolderWithPaths(datasets.ImageFolder):
    """画像フォルダのパスを保持するデータセットクラス

    Attributes:
        flip_label_prob: ラベルを反転する確率
    """

    def __init__(
        self,
        path: str | os.PathLike,
        transform: torchvision.transforms.Compose,
        flip_label_prob: float = 0.0
    ) -> None:
        """画像フォルダのパスを保持するデータセットクラスを初期化

        Args:
            path: 画像フォルダのパス
            transform: 前処理関数
            flip_label_prob: ラベルを反転する確率. Defaults to 0.0.
        """
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f"Flipping labels with probability {self.flip_label_prob}")
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes-1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index) -> Dict[str, int, str | os.PathLike]:
        """指定したインデックスのデータを取得

        Args:
            index: インデックス

        Returns:
            画像データ、ラベル、画像のパスを含む辞書
        """
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            "images": image,
            "labels": label,
            "image_paths": self.samples[index][0]
        }


def maybe_dictionarize(
    batch
) -> Dict[torch.Tensor, int] | Dict[torch.Tensor, int, Any]:
    """バッチを辞書に変換

    Args:
        batch: 変換対象のバッチ

    Raises:
        ValueError: 要素数が想定外の場合

    Returns:
        辞書形式に変換されたバッチ
    """
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {"images": batch[0], "labels": batch[1]}
    elif len(batch) == 3:
        batch = {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")

    return batch


def get_features_helper(
    image_encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """データローダーから特徴量を取得

    Args:
        image_encoder: 特徴量抽出用のエンコーダ
        dataloader: 入力データのローダー
        device: 計算に使用するデバイス

    Returns:
        抽出された特徴量を含む辞書
    """
    all_data = collections.defaultdict(list)

    # エンコーダの設定
    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(
        image_encoder,
        device_ids=[x for x in range(torch.cuda.device_count())]
    )
    image_encoder.eval()

    # 特徴量の抽出
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch["images"].cuda())

            all_data["features"].append(features.cpu())

            for key, val in batch.items():
                if key == "images":
                    continue
                if hasattr(val, "cpu"):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    # テンソルの結合
    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(
    is_train: bool,
    image_encoder: nn.Module,
    dataset: Any,   # データセットのラッパークラス
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """キャッシュを考慮して特徴量を取得

    Args:
        is_train: 学習データかどうか
        image_encoder: 特徴量抽出用のエンコーダ
        dataset: 入力データセット
        device: 計算に使用するデバイス

    Returns:
        特徴量を含む辞書
    """
    split = "train" if is_train else "val"
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f"{image_encoder.cache_dir}/{dname}/{split}"
        cached_files = glob.glob(f"{cache_dir}/*")
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f"Getting features from {cache_dir}")
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(
            f"Did not find cached features at {cache_dir}. "
            f"Building from scratch."
        )
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print("Not caching because no cache directory was passed.")
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching data at {cache_dir}")
            for name, val in data.items():
                torch.save(val, f"{cache_dir}/{name}.pt")
    return data


class FeatureDataset(Dataset):
    """特徴量を保持するデータセットクラス

    Attributes:
        data: 特徴量データ
    """

    def __init__(
        self,
        is_train: bool,
        image_encoder: nn.Module,
        dataset: Dataset,
        device: torch.device
    ) -> None:
        """特徴量を保持するデータセットクラスを初期化

        Args:
            is_train: 学習データかどうか
            image_encoder: 特徴量抽出用のエンコーダ
            dataset: 入力データセット
            device: 計算に使用するデバイス
        """
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self) -> int:
        """データの数を取得

        Returns:
            データの数
        """
        return len(self.data["features"])

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """指定したインデックスのデータを取得

        Args:
            idx: インデックス

        Returns:
            特徴量データを含む辞書
        """
        data = {k: v[idx] for k, v in self.data.items()}
        data["features"] = torch.from_numpy(data["features"]).float()
        return data


def get_dataloader(
    dataset: Any,   # データセットのラッパークラス
    is_train: bool,
    args: Namespace,
    image_encoder: nn.Module = None
) -> DataLoader | Any:
    """データローダーを取得

    Args:
        dataset: 入力データセット
        is_train: 学習データかどうか
        args: 設定パラメータ
        image_encoder: 特徴量抽出用のエンコーダ. Defaults to None.

    Returns:
        データローダー
    """
    if image_encoder is not None:
        feature_dataset = FeatureDataset(
            is_train, image_encoder,
            dataset, args.device
        )
        dataloader = DataLoader(
            feature_dataset,
            batch_size=args.batch_size, shuffle=is_train
        )
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader

# CIFAR-100データセットを扱うためのクラスを定義
import os

import torch
import torchvision
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100


class CIFAR100:
    """
    CIFAR-100データセットのラッパークラス

    Attributes:
        train_dataset (PyTorchCIFAR100): 学習用データセット
        train_loader (torch.utils.data.DataLoader): 学習用データローダー
        test_dataset (PyTorchCIFAR100): テスト用データセット
        test_loader (torch.utils.data.DataLoader): テスト用データローダー
        classnames (List[str]): クラス名のリスト
    """
    def __init__(self,
                 preprocess: torchvision.transforms.Compose,
                 location: str | os.PathLike = os.path.expanduser("dataset"),
                 batch_size: int = 32,
                 num_workers: int = 4) -> None:
        """
        CIFAR-100データセットを扱うクラスを初期化

        Args:
            preprocess (torchvision.transforms.Compose): 前処理関数
            location (str | os.PathLike, optional): データセットの保存先ディレクトリ. \
                Defaults to os.path.expanduser("dataset").
            batch_size (int, optional): バッチサイズ. Defaults to 32.
            num_workers (int, optional): データローダーの並列数. Defaults to 4.
        """
        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = CIFAR100(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

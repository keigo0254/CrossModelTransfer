# DTDのデータセットを扱うためのクラス
import os

import torch
import torchvision
import torchvision.datasets as datasets


class DTD:
    """
    DTDのデータセットを扱うためのラッパークラス

    Attributes:
        train_dataset (datasets.ImageFolder): 学習用データセット
        train_loader (torch.utils.data.DataLoader): 学習用データローダー
        test_dataset (datasets.ImageFolder): テスト用データセット
        test_loader (torch.utils.data.DataLoader): テスト用データローダー
        classnames (List[str]): クラス名のリスト
    """
    def __init__(
        self,
        preprocess: torchvision.transforms.Compose,
        location: str | os.PathLike = os.path.expanduser("dataset"),
        batch_size: int = 32,
        num_workers: int = 4
    ) -> None:
        """
        DTDのデータセットを扱うクラスを初期化

        Args:
            preprocess (torchvision.transforms.Compose): 前処理関数
            location (str | os.PathLike, optional): データセットの保存先ディレクトリ. \
                Defaults to os.path.expanduser("dataset").
            batch_size (int, optional): バッチサイズ. Defaults to 32.
            num_workers (int, optional): データローダーの並列数. Defaults to 4.
        """
        # Data loading code
        traindir = os.path.join(location, "dtd", "train")
        valdir = os.path.join(location, "dtd", "val")

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            "_", " ") for i in range(len(idx_to_class))]


if __name__ == "__main__":
    # 動作検証
    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    root = os.path.expanduser("dataset")
    d = DTD(preprocess, location=root)
    for i, (data, target) in enumerate(d.train_loader):
        print(data.shape, target)
        break

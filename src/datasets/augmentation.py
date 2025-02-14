# データ拡張を行う
import random
from typing import List

from PIL import ImageFilter, ImageOps, Image
import numpy as np
import torchvision.transforms as transforms


class gray_scale(object):
    """
    グレースケール変換

    Attributes:
        p (float): グレースケール変換を行う確率
        transf (transforms.Grayscale): グレースケール変換を行う関数
    """
    def __init__(self, p: float = 0.2) -> None:
        """
        グレースケール変換を行う関数を初期化

        Args:
            p (float, optional): グレースケール変換を行う確率. Defaults to 0.2.
        """
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img: Image) -> Image:
        """
        グレースケール変換を行う

        Args:
            img (Image): 変換する画像

        Returns:
            Image: 変換後の画像
        """
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class horizontal_flip(object):
    """
    水平方向に反転する関数

    Attributes:
        p (float): 反転する確率
        transf (transforms.RandomHorizontalFlip): 反転を行う関数
    """
    def __init__(self, p: float = 0.2, activate_pred: bool = False) -> None:
        """
        水平方向に反転する関数を初期化

        Args:
            p (float, optional): 反転する確率. Defaults to 0.2.
            activate_pred (bool, optional): 推論時にも反転するかどうか. Defaults to False.
        """
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, img: Image) -> Image:
        """
        水平方向に反転する

        Args:
            img (Image): 変換する画像

        Returns:
            Image: 変換後の画像
        """
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class Solarization(object):
    """
    ソラリゼーション

    Attributes:
        p (float): ソラリゼーションを行う確率
    """
    def __init__(self, p: float = 0.2) -> None:
        """
        ソラリゼーションを行う関数を初期化

        Args:
            p (float, optional): ソラリゼーションを行う確率. Defaults to 0.2.
        """
        self.p = p

    def __call__(self, img: Image) -> Image:
        """
        ソラリゼーションを行う

        Args:
            img (Image): 変換する画像

        Returns:
            Image: 変換後の画像
        """
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):
    """
    ガウシアンブラー

    Attributes:
        prob (float): ガウシアンブラーを行う確率
        radius_min (float): ガウシアンブラーの最小半径
        radius_max (float): ガウシアンブラーの最大半径
    """
    def __init__(
        self, p: float = 0.1, radius_min: float = 0.1, radius_max: float = 2.
    ) -> None:
        """
        ガウシアンブラーを行う関数を初期化

        Args:
            p (float, optional): ガウシアンブラーを行う確率. Defaults to 0.1.
            radius_min (float, optional): ガウシアンブラーの最小半径. Defaults to 0.1.
            radius_max (float, optional): ガウシアンブラーの最大半径. Defaults to 2.
        """
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image) -> Image:
        """
        ガウシアンブラーを行う

        Args:
            img (Image): 変換する画像

        Returns:
            Image: 変換後の画像
        """
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


def get_augmented_preprocess_fn(
    preprocess: transforms,
    p: float = 1.0
) -> transforms.Compose[List]:
    """
    データ拡張を行う前処理関数を取得する

    Args:
        preprocess (transforms): ベースとなる前処理関数．\
        open_clip.create_model_and_transformsで取得したものを指定する
        p (float, optional): データ拡張を行う確率. Defaults to 1.0.

    Returns:
        transforms.Compose[List]: データ拡張を行う前処理関数
    """
    mean = np.array(preprocess.transforms[3].mean)
    std = np.array(preprocess.transforms[3].std)

    mean2 = -1 * mean / std
    std2 = 1 / std

    return transforms.Compose([
        *preprocess.transforms,
        transforms.Normalize(mean2.tolist(), std2.tolist()),
        transforms.ToPILImage(),
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            gray_scale(p),
            Solarization(p),
            GaussianBlur(p)
        ]),
        transforms.ToTensor(),
        preprocess.transforms[-1]
    ])


if __name__ == "__main__":
    # 動作確認用
    import os

    import open_clip

    root = os.path.expanduser("dataset")

    _, original_preprocess_fn, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai", cache_dir=".cache"
    )

    augmented_preprocess_fn = get_augmented_preprocess_fn(
        original_preprocess_fn)
    print(augmented_preprocess_fn)

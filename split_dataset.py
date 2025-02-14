"""
データセットを分割するモジュール

データセットのディレクトリをtrain, val(, test)に分割するためのスクリプトを提供する。
SUN397, EuroSAT, DTD, ImageNetの各データセットに対応している。

Original Code:
https://github.com/mlfoundations/task_vectors/issues/1

Functions:
    process_dataset: SUN397/DTDデータセットの画像を指定されたディレクトリにコピーする
    create_directory_structure: EuroSATデータセットのディレクトリ構造を作成する
    split_dataset: EuroSATデータセットをtrain, val, testに分割して保存する
"""

import glob
import os
import random
import shutil
from typing import List
import scipy.io
import tarfile


# データセットの保存先ディレクトリ(環境に合わせて変更する)
base_dir = os.path.expanduser("~/dataset")


def process_dataset(
    txt_file: str | os.PathLike,
    downloaded_data_path: str | os.PathLike,
    output_folder: str | os.PathLike
) -> None:
    """SUN397データセットの画像を指定されたディレクトリにコピーする

    Args:
        txt_file: 画像パスが記載されたテキストファイルのパス
        downloaded_data_path: ダウンロードしたデータセットのパス
        output_folder: 出力先のディレクトリパス
    """
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split('/')[:-1])[1:]
        filename = input_path.split('/')[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path[1:])
        output_file_path = os.path.join(output_class_folder, filename)

        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")


# SUN397データセットの処理
downloaded_data_path = f"{base_dir}/sun397"
output_path = f"{base_dir}/sun397"

process_dataset(
    os.path.join(downloaded_data_path, 'Training_01.txt'),
    os.path.join(downloaded_data_path, 'SUN397'),
    os.path.join(output_path, "train")
)
process_dataset(
    os.path.join(downloaded_data_path, 'Testing_01.txt'),
    os.path.join(downloaded_data_path, 'SUN397'),
    os.path.join(output_path, "val")
)


def create_directory_structure(
    dst_dir: str | os.PathLike,
    classes: List[str]
) -> None:
    """EuroSATデータセットのディレクトリ構造を作成する

    Args:
        dst_dir: データセットの保存先ディレクトリ
        classes: クラス名のリスト
    """
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(dst_dir, dataset)
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)


def split_dataset(
    dst_dir: str | os.PathLike,
    src_dir: str | os.PathLike,
    classes: List[str],
    val_size: int = 270,
    test_size: int = 270
) -> None:
    """EuroSATデータセットをtrain, val, testに分割して保存する

    Args:
        dst_dir: データセットの保存先ディレクトリ
        src_dir: データセットの元ディレクトリ
        classes: クラス名のリスト
        val_size: valデータのサイズ. Defaults to 270.
        test_size: testデータのサイズ. Defaults to 270.
    """
    for cls in classes:
        class_path = os.path.join(src_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dst_dir, 'train', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dst_dir, 'val', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dst_dir, 'test', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)


# EuroSATデータセットの処理
src_dir = f'{base_dir}/euro_sat/2750'
dst_dir = f'{base_dir}/EuroSAT_splits'

classes = [
    d for d in os.listdir(src_dir)
    if os.path.isdir(os.path.join(src_dir, d))
]
create_directory_structure(dst_dir, classes)
split_dataset(dst_dir, src_dir, classes)


def process_dataset(
    txt_file: str | os.PathLike,
    downloaded_data_path: str | os.PathLike,
    output_folder: str | os.PathLike
) -> None:
    """DTDデータセットを処理して、指定されたフォルダに保存する

    Args:
        txt_file: 分割を記述したテキストファイルのパス
        downloaded_data_path: ダウンロードしたデータセットのパス
        output_folder: 処理したデータセットを保存するフォルダのパス
    """
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = input_path.split('/')[:-1][0]
        filename = input_path.split('/')[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path)
        output_file_path = os.path.join(output_class_folder, filename)
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")


# DTDデータセットの処理
downloaded_data_path = f"{base_dir}/dtd/images"
output_path = f"{base_dir}/dtd"

process_dataset(
    f'{base_dir}/dtd/labels/train.txt',
    downloaded_data_path,
    os.path.join(output_path, "train")
)
process_dataset(
    f'{base_dir}/dtd/labels/test.txt',
    downloaded_data_path,
    os.path.join(output_path, "val")
)


# ImageNetデータセットの処理
# Original Code:
# https://zenn.dev/hidetoshi/articles/20210717_pytorch_dataset_for_imagenet

# trainデータセットの処理
target_dir = f"{base_dir}/ILSVRC2012_img_train/"
target_dir = shutil.move(target_dir, f"{base_dir}/imagenet/train")
for tar_filepath in glob.glob(os.path.join(target_dir, "*.tar")):
    target_dir = tar_filepath.replace(".tar", "")
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(tar_filepath, "r") as tar:
        tar.extractall(target_dir)
    os.remove(tar_filepath)
os.remove(f"{base_dir}/ILSVRC2012_img_train.tar")

# valデータセットの処理
imagenet_val_tar_path = f"{base_dir}/ILSVRC2012_img_val.tar"
target_dir = f"{base_dir}/imagenet/val_in_folder/"
meta_path = f"{base_dir}/ILSVRC2012_devkit_t12/data/meta.mat"
truth_label_path = (
    f"{base_dir}/ILSVRC2012_devkit_t12/"
    f"data/ILSVRC2012_validation_ground_truth.txt"
)

meta = scipy.io.loadmat(meta_path, squeeze_me=True)
ilsvrc2012_id_to_wnid = {m[0]: m[1] for m in meta["synsets"]}

with open(truth_label_path, "r") as f:
    ilsvrc_ids = tuple(
        int(ilsvrc_id) for ilsvrc_id in f.read().split("\n")[:-1]
    )

for ilsvrc_id in ilsvrc_ids:
    wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
    os.makedirs(os.path.join(target_dir, wnid), exist_ok=True)

os.makedirs(target_dir, exist_ok=True)

num_valid_images = 50000
with tarfile.open(imagenet_val_tar_path, "r") as tar:
    for valid_id, ilsvrc_id in zip(range(1, num_valid_images+1), ilsvrc_ids):
        wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
        filename = f"ILSVRC2012_val_{str(valid_id).zfill(8)}.JPEG"
        print(filename, wnid)
        img = tar.extractfile(filename)
        with open(os.path.join(target_dir, wnid, filename), "wb") as f:
            f.write(img.read())

# 不要なファイルの削除
os.remove(f"{base_dir}/ILSVRC2012_img_val.tar")
shutil.rmtree(f"{base_dir}/ILSVRC2012_devkit_t12")
os.remove(f"{base_dir}/ILSVRC2012_devkit_t12.tar.gz")

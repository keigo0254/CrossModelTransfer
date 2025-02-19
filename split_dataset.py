"""
Original Code:
https://github.com/mlfoundations/task_vectors/issues/1
"""

import glob
import os
import random
import shutil
from typing import List
import scipy.io
import tarfile


# Modify this path to your dataset directory
base_dir = os.path.expanduser("~/dataset")


# Process SUN397 dataset
def process_dataset(
    txt_file: str | os.PathLike,
    downloaded_data_path: str | os.PathLike,
    output_folder: str | os.PathLike
) -> None:
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


# Process EuroSAT dataset
def create_directory_structure(
    dst_dir: str | os.PathLike,
    classes: List[str]
) -> None:
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


src_dir = f'{base_dir}/euro_sat/2750'
dst_dir = f'{base_dir}/EuroSAT_splits'

classes = [
    d for d in os.listdir(src_dir)
    if os.path.isdir(os.path.join(src_dir, d))
]
create_directory_structure(dst_dir, classes)
split_dataset(dst_dir, src_dir, classes)


# Process DTD dataset
def process_dataset(
    txt_file: str | os.PathLike,
    downloaded_data_path: str | os.PathLike,
    output_folder: str | os.PathLike
) -> None:
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


# Process ImageNet dataset
# Original Code:
# https://zenn.dev/hidetoshi/articles/20210717_pytorch_dataset_for_imagenet

target_dir = f"{base_dir}/ILSVRC2012_img_train/"
target_dir = shutil.move(target_dir, f"{base_dir}/imagenet/train")
for tar_filepath in glob.glob(os.path.join(target_dir, "*.tar")):
    target_dir = tar_filepath.replace(".tar", "")
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(tar_filepath, "r") as tar:
        tar.extractall(target_dir)
    os.remove(tar_filepath)
os.remove(f"{base_dir}/ILSVRC2012_img_train.tar")

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

os.remove(f"{base_dir}/ILSVRC2012_img_val.tar")
shutil.rmtree(f"{base_dir}/ILSVRC2012_devkit_t12")
os.remove(f"{base_dir}/ILSVRC2012_devkit_t12.tar.gz")

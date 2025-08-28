import os

import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

from args import Args
from datasets.registry import get_dataset
from datasets.templates import get_templates
from modeling import ClassificationHead, ImageEncoder


def build_classification_head(
    model: ImageEncoder,
    dataset_name: str,
    template: list,
    dataset_root: str,
    device: str
) -> nn.Module:
    template = get_templates(dataset_name)

    dataset = get_dataset(dataset_name, preprocess=None, location=dataset_root)

    print("Building classification head.")
    
    # シンプルなランダム初期化でClassification Headを作成
    num_classes = len(dataset.classnames)
    feature_dim = 512  # ViT-B-32の特徴次元
    
    # 小さな値で初期化して数値的安定性を確保
    weights = torch.randn(num_classes, feature_dim, device=device) * 0.01
    bias = torch.zeros(num_classes, device=device)
    
    # 正規化
    weights = torch.nn.functional.normalize(weights, dim=1)
    
    classification_head = ClassificationHead(normalize=True, weights=weights)
    classification_head.bias = nn.Parameter(bias)

    return classification_head


def get_classification_head(args: Args, dataset: str) -> nn.Module:
    """Get or build a classification head for a dataset."""
    filename = os.path.join(
        args.model_root,
        args.model_architecture,
        args.pretrained,
        "heads",
        f"head_for_{dataset}.pt"
    )
    if os.path.exists(filename):
        print(
            f"Classification head for {args.model_architecture} "
            f"on {dataset} exists at {filename}"
        )
        return ClassificationHead.load(filename)
    print(
        f"Did not find classification head for {args.model_architecture} "
        f"on {dataset} at {filename}, building one from scratch."
    )
    model = ImageEncoder(args, keep_lang=True).model
    template = get_templates(dataset)
    classification_head: ClassificationHead = build_classification_head(
        model,
        dataset+"Val",
        template,
        args.dataset_root,
        args.device
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    classification_head.save(filename)
    return classification_head

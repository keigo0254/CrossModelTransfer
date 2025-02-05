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

    logit_scale = model.logit_scale
    dataset = get_dataset(dataset_name, preprocess=None, location=dataset_root)

    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)    # tokenize
            embeddings = model.encode_text(texts)           # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(args: Args, dataset: str) -> nn.Module:
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
        dataset,
        template,
        args.dataset_root,
        args.device
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    classification_head.save(filename)
    return classification_head


if __name__ == "__main__":
    args: Args = Args().from_args()
    args.train_datasets = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN",
        "CIFAR10", "CIFAR100", "ImageNet", "STL10"
    ]

    for dataset in args.train_datasets:
        classification_head = get_classification_head(args, dataset)
        print(classification_head)
    
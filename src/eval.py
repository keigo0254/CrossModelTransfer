import copy
import json
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from args import Args
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.mixed_dataset import MixedDataset
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder, MultiHeadImageClassifier
import utils


def eval_multihead_classifier(
    classifier: MultiHeadImageClassifier,
    args: Args
) -> float:
    """Evaluate a multi-head classifier on a mixed dataset."""
    dataset = MixedDataset(
        args.eval_datasets,
        args.dataset_root,
        num_images=args.num_images,
        preprocess=classifier.val_preprocess
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    device = args.device
    classifier.eval()
    classifier = classifier.to(device)

    with torch.no_grad():
        correct, n = 0, 0
        for _, batch in enumerate(tqdm(dataloader)):
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(device)
            y = batch["labels"].to(device)
            y = list(y.squeeze())
            dataset_name = batch["metadata"]

            # Get predictions for each dataset head
            logits = classifier(x)
            for i, (label, name) in enumerate(zip(y, dataset_name)):
                pred = logits[name][i].argmax(dim=0, keepdim=True).to(device)
                correct += pred.eq(label.view_as(pred)).sum().item()
                n += 1

        top1 = correct / n

    print(f"Done evaluating on mixed dataset. Accuracy: {top1:.2%}")

    return top1


def eval_single_dataset(
    image_encoder: ImageEncoder,
    dataset_name: str,
    args: Args
) -> float:
    """Evaluate a single dataset."""
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.dataset_root,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset,
        is_train=False,
        args=args,
        image_encoder=None
    )

    device = args.device
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        correct, n = 0, 0
        for _, batch in enumerate(tqdm(dataloader)):
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(device)
            y = batch["labels"].to(device)

            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n

    print(f"Done evaluating on {dataset_name}. Accuracy: {top1:.2%}")

    return top1


def evaluate(image_encoder: ImageEncoder, args: Args) -> Dict[str, float]:
    """Evaluate the model on multiple datasets."""
    if args.eval_datasets is None:
        print("No dataset to evaluate on.")
        return

    info = {}

    # Evaluate on each dataset
    for _, dataset_name in enumerate(args.eval_datasets):
        print(f"\nEvaluating on {dataset_name}\n")
        top1 = eval_single_dataset(image_encoder, dataset_name, args)
        info[dataset_name] = top1
        print(f"{dataset_name} Top-1 accuracy: {top1:.2%}")

    # Calculate average accuracy
    info["AVG."] = np.mean([info[dataset_name] for dataset_name in args.eval_datasets])

    # Save results if specified
    if args.result is not None:
        os.makedirs(os.path.dirname(args.result), exist_ok=True)
        with open(args.result, "w") as f:
            json.dump(info, f, indent=4)
        print(f"result saved to {args.result}")

    # Create visualization if specified
    if args.fig is not None:
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        dataset_names = copy.deepcopy(args.eval_datasets)
        dataset_names.append("AVG.")

        accuracies = [info[dataset_name] * 100 for dataset_name in dataset_names]

        fig, ax = plt.subplots()
        ax.bar(dataset_names, accuracies, width=0.4, tick_label=dataset_names)
        ax.set_ylim(0, 105)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, fontsize=10)
        for x, y in zip(dataset_names, accuracies):
            plt.text(x, y, f"{y:.4}", ha='center', va='bottom')
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Dataset")
        ax.set_title("Accuracy")

        fig.savefig(args.fig)
        plt.close(fig)

    return info

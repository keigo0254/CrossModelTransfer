import json
import os
import random
from typing import Callable, Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from args import Args
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.mixed_dataset import MixedDataset
from datasets.registry import get_dataset
from modeling import ImageEncoder
from task_vectors import TaskVector


def set_feature_hook(
    name: str, features: Dict[str, torch.Tensor] = {}
) -> Callable:
    """Set a feature hook for a given name"""
    def hook(module, input, output):
        features[name] = output
    return hook


def get_inner_features(
    image_encoder: ImageEncoder,
    dataset: Dataset,
    args: Args
) -> Dict[str, torch.Tensor]:
    """Get the inner features of the image encoder"""
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_encoder)

    features = {}
    print("Setting feature hooks:")
    for key, module in image_encoder.named_modules():
        print(key)
        module.register_forward_hook(set_feature_hook(key, features))

    image_encoder = image_encoder.to(args.device)
    image_encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(args.device)
            _ = image_encoder(images)

    return features


def inspect_features(features: Dict[str, torch.Tensor], args: Args) -> None:
    """Inspect the features of the image encoder"""
    pre_q, pre_k, pre_v, pre_out = [], [], [], []
    delta_q, delta_k, delta_v, delta_out = [], [], [], []
    for key, feature in features.items():
        if "q_proj" in key:
            pass


if __name__ == "__main__":
    args: Args = Args().from_args()
    if args.finetuning_type != "lora":
        args.rank = 0
        args.alpha = 0

    SEED = args.seed
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    pretrained_encoder = ImageEncoder(args, keep_lang=False)

    task_vector = TaskVector.load_vector(os.path.join(
        args.model_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        "task_vector",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"task_vector_for_{args.eval_datasets}.pt"
    ))

    image_encoder = task_vector.apply_to(pretrained_encoder, args.lamb)
    features = get_inner_features(image_encoder, args.eval_datasets, args)


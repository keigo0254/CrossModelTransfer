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


def inspect_features(
    features: Dict[str, torch.Tensor], args: Args, num_layers: int = 12
) -> None:
    """Inspect the features of the image encoder"""
    pre_q, pre_k, pre_v, pre_out = [], [], [], []
    delta_q, delta_k, delta_v, delta_out = [], [], [], []
    keys = [
        f"model.visual.transformer.resblocks.{i}.attn"
        for i in range(num_layers)
    ]
    for key in keys:
        pre_q.append(features[key + ".q_proj.weight"])
        pre_k.append(features[key + ".k_proj.weight"])
        pre_v.append(features[key + ".v_proj.weight"])
        pre_out.append(features[key + ".out_proj.weight"])
        delta_q.append(features[key + ".q_proj.Delta"])
        delta_k.append(features[key + ".k_proj.Delta"])
        delta_v.append(features[key + ".v_proj.Delta"])
        delta_out.append(features[key + ".out_proj.Delta"])

    pre_q_norm = [torch.linalg.matrix_norm(q, p="fro") for q in pre_q]
    delta_q_norm = [torch.linalg.matrix_norm(q, p="fro") for q in delta_q]
    pre_k_norm = [torch.linalg.matrix_norm(k, p="fro") for k in pre_k]
    delta_k_norm = [torch.linalg.matrix_norm(k, p="fro") for k in delta_k]
    pre_v_norm = [torch.linalg.matrix_norm(v, p="fro") for v in pre_v]
    delta_v_norm = [torch.linalg.matrix_norm(v, p="fro") for v in delta_v]
    pre_out_norm = [torch.linalg.matrix_norm(out, p="fro") for out in pre_out]
    delta_out_norm = [torch.linalg.matrix_norm(out, p="fro") for out in delta_out]

    fig, ax = plt.subplots()
    ax.plot(pre_q_norm, label="pre_q", color="red", linestyle="--")
    ax.plot(delta_q_norm, label="delta_q", color="red")
    ax.plot(pre_k_norm, label="pre_k", color="blue", linestyle="--")
    ax.plot(delta_k_norm, label="delta_k", color="blue")
    ax.plot(pre_v_norm, label="pre_v", color="green", linestyle="--")
    ax.plot(delta_v_norm, label="delta_v", color="green")
    ax.plot(pre_out_norm, label="pre_out", color="orange", linestyle="--")
    ax.plot(delta_out_norm, label="delta_out", color="orange")
    ax.legend()
    ax.set_title("Feature Norms")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Norm")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        "feature",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"feature_norms_{args.eval_datasets}.jpg"
    )
    plt.savefig(filename)
    plt.close()


def inspect_weights(image_encoder: ImageEncoder, args: Args) -> None:
    """Inspect the weights of the image encoder"""
    pre_q, pre_k, pre_v, pre_out = [], [], [], []
    delta_q, delta_k, delta_v, delta_out = [], [], [], []
    keys = [
        f"model.visual.transformer.resblocks.{i}.attn"
        for i in range(image_encoder.model.visual.transformer.layers)
    ]
    state_dict = image_encoder.state_dict()
    for key in keys:
        pre_q.append(state_dict[key + ".q_proj.weight"])
        pre_k.append(state_dict[key + ".k_proj.weight"])
        pre_v.append(state_dict[key + ".v_proj.weight"])
        pre_out.append(state_dict[key + ".out_proj.weight"])
        if args.finetuning_type == "lora":
            delta_q.append(state_dict[key + ".q_proj.Delta.B"] @ state_dict[key + ".q_proj.Delta.A"])
            delta_k.append(state_dict[key + ".k_proj.Delta.B"] @ state_dict[key + ".k_proj.Delta.A"])
            delta_v.append(state_dict[key + ".v_proj.Delta.B"] @ state_dict[key + ".v_proj.Delta.A"])
            delta_out.append(state_dict[key + ".out_proj.Delta.B"] @ state_dict[key + ".out_proj.Delta.A"])
        else:
            delta_q.append(state_dict[key + ".q_proj.Delta.D"])
            delta_k.append(state_dict[key + ".k_proj.Delta.D"])
            delta_v.append(state_dict[key + ".v_proj.Delta.D"])
            delta_out.append(state_dict[key + ".out_proj.Delta.D"])

    pre_q_norm = [torch.linalg.matrix_norm(q, p="fro") for q in pre_q]
    delta_q_norm = [torch.linalg.matrix_norm(q, p="fro") for q in delta_q]
    pre_k_norm = [torch.linalg.matrix_norm(k, p="fro") for k in pre_k]
    delta_k_norm = [torch.linalg.matrix_norm(k, p="fro") for k in delta_k]
    pre_v_norm = [torch.linalg.matrix_norm(v, p="fro") for v in pre_v]
    delta_v_norm = [torch.linalg.matrix_norm(v, p="fro") for v in delta_v]
    pre_out_norm = [torch.linalg.matrix_norm(out, p="fro") for out in pre_out]
    delta_out_norm = [torch.linalg.matrix_norm(out, p="fro") for out in delta_out]

    fig, ax = plt.subplots()
    ax.plot(pre_q_norm, label="pre_q", color="red", linestyle="--")
    ax.plot(delta_q_norm, label="delta_q", color="red")
    ax.plot(pre_k_norm, label="pre_k", color="blue", linestyle="--")
    ax.plot(delta_k_norm, label="delta_k", color="blue")
    ax.plot(pre_v_norm, label="pre_v", color="green", linestyle="--")
    ax.plot(delta_v_norm, label="delta_v", color="green")
    ax.plot(pre_out_norm, label="pre_out", color="orange", linestyle="--")
    ax.plot(delta_out_norm, label="delta_out", color="orange")
    ax.legend()
    ax.set_title("Feature Norms")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Norm")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        "feature",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"weight_norms_{args.eval_datasets}.jpg"
    )
    plt.savefig(filename)
    plt.close()


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

    image_encoder: ImageEncoder = task_vector.apply_to(pretrained_encoder, args.lamb)
    features = get_inner_features(image_encoder, args.eval_datasets, args)
    inspect_features(features, args, image_encoder.model.visual.transformer.layers)
    inspect_weights(image_encoder, args)

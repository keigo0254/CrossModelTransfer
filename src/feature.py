import math
import os
import random
from typing import Callable, Dict, Tuple

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import Dataset
from args import Args
from datasets.common import get_dataloader, maybe_dictionarize
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
    dataloader = get_dataloader(dataset, is_train=False, args=args)

    features = {}
    for key, module in image_encoder.named_modules():
        module.register_forward_hook(set_feature_hook(key, features))

    image_encoder = image_encoder.to(args.device)
    image_encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(args.device)
            _ = image_encoder(images)

    return features


def find_best_grid(n) -> Tuple[int, int]:
    """Find the best grid for n"""
    factors = [(i, n // i) for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0]
    return min(factors, key=lambda x: abs(x[0] - x[1]))[::-1]


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
        pre_q.append(features[key + ".q_proj.Pre"])
        pre_k.append(features[key + ".k_proj.Pre"])
        pre_v.append(features[key + ".v_proj.Pre"])
        pre_out.append(features[key + ".out_proj.Pre"])
        delta_q.append(features[key + ".q_proj.Delta"])
        delta_k.append(features[key + ".k_proj.Delta"])
        delta_v.append(features[key + ".v_proj.Delta"])
        delta_out.append(features[key + ".out_proj.Delta"])

    embed_dim = pre_q[0].shape[-1]

    pre_q_norm = [
        [
            torch.linalg.matrix_norm(q_patch, ord="fro").to("cpu").numpy()
            for q_patch in q_layer
        ]
        for q_layer in pre_q
    ]
    pre_q_norm_cls_token = [pre_q_norm[i][0] for i in range(len(pre_q_norm))]
    for i in range(len(pre_q_norm)):
        pre_q_norm[i].pop(0)
    pre_q_norm_reshaped = np.array(pre_q_norm).reshape(len(pre_q_norm), int(math.sqrt(len(pre_q_norm[0]))), -1)
    delta_q_norm = [
        [
            torch.linalg.matrix_norm(q_patch, ord="fro").to("cpu").numpy()
            for q_patch in q_layer
        ]
        for q_layer in delta_q
    ]
    delta_q_norm_cls_token = [delta_q_norm[i][0] for i in range(len(delta_q_norm))]
    for i in range(len(delta_q_norm)):
        delta_q_norm[i].pop(0)
    delta_q_norm_reshaped = np.array(delta_q_norm).reshape(len(delta_q_norm), int(math.sqrt(len(delta_q_norm[0]))), -1)

    best_rows, best_cols = find_best_grid(len(pre_q_norm))
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(pre_q_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "q_pre_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(delta_q_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "q_delta_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    pre_k_norm = [
        [
            torch.linalg.matrix_norm(k_patch, ord="fro").to("cpu").numpy()
            for k_patch in k_layer
        ]
        for k_layer in pre_k
    ]
    pre_k_norm_cls_token = [pre_k_norm[i][0] for i in range(len(pre_k_norm))]
    for i in range(len(pre_k_norm)):
        pre_k_norm[i].pop(0)
    pre_k_norm_reshaped = np.array(pre_k_norm).reshape(len(pre_k_norm), int(math.sqrt(len(pre_k_norm[0]))), -1)
    delta_k_norm = [
        [
            torch.linalg.matrix_norm(k_patch, ord="fro").to("cpu").numpy()
            for k_patch in k_layer
        ]
        for k_layer in delta_k
    ]
    delta_k_norm_cls_token = [delta_k_norm[i][0] for i in range(len(delta_k_norm))]
    for i in range(len(delta_k_norm)):
        delta_k_norm[i].pop(0)
    delta_k_norm_reshaped = np.array(delta_k_norm).reshape(len(delta_k_norm), int(math.sqrt(len(delta_k_norm[0]))), -1)

    best_rows, best_cols = find_best_grid(len(pre_k_norm))
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(pre_k_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "k_pre_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(delta_k_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "k_delta_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    pre_v_norm = [
        [
            torch.linalg.matrix_norm(v_patch, ord="fro").to("cpu").numpy()
            for v_patch in v_layer
        ]
        for v_layer in pre_v
    ]
    pre_v_norm_cls_token = [pre_v_norm[i][0] for i in range(len(pre_v_norm))]
    for i in range(len(pre_v_norm)):
        pre_v_norm[i].pop(0)
    pre_v_norm_reshaped = np.array(pre_v_norm).reshape(len(pre_v_norm), int(math.sqrt(len(pre_v_norm[0]))), -1)
    delta_v_norm = [
        [
            torch.linalg.matrix_norm(v_patch, ord="fro").to("cpu").numpy()
            for v_patch in v_layer
        ]
        for v_layer in delta_v
    ]
    delta_v_norm_cls_token = [delta_v_norm[i][0] for i in range(len(delta_v_norm))]
    for i in range(len(delta_v_norm)):
        delta_v_norm[i].pop(0)
    delta_v_norm_reshaped = np.array(delta_v_norm).reshape(len(delta_v_norm), int(math.sqrt(len(delta_v_norm[0]))), -1)

    best_rows, best_cols = find_best_grid(len(pre_v_norm))
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(pre_v_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "v_pre_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(delta_v_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "v_delta_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    for i in range(len(pre_out)):
        pre_out[i] = pre_out[i].to("cpu").reshape(len(delta_v_norm[0]) + 1, -1, embed_dim)
        delta_out[i] = delta_out[i].to("cpu").reshape(len(delta_v_norm[0]) + 1, -1, embed_dim)

    pre_out_norm = [
        [
            torch.linalg.matrix_norm(out_patch, ord="fro").to("cpu").numpy()
            for out_patch in out_layer
        ]
        for out_layer in pre_out
    ]

    pre_out_norm_cls_token = [pre_out_norm[i][0] for i in range(len(pre_out_norm))]
    for i in range(len(pre_out_norm)):
        pre_out_norm[i].pop(0)
    pre_out_norm_reshaped = np.array(pre_out_norm).reshape(len(pre_out_norm), int(math.sqrt(len(pre_out_norm[0]))), -1)
    delta_out_norm = [
        [
            torch.linalg.matrix_norm(out_patch, ord="fro").to("cpu").numpy()
            for out_patch in out_layer
        ]
        for out_layer in delta_out
    ]
    delta_out_norm_cls_token = [delta_out_norm[i][0] for i in range(len(delta_out_norm))]
    for i in range(len(delta_out_norm)):
        delta_out_norm[i].pop(0)
    delta_out_norm_reshaped = np.array(delta_out_norm).reshape(len(delta_out_norm), int(math.sqrt(len(delta_out_norm[0]))), -1)

    best_rows, best_cols = find_best_grid(len(pre_out_norm))
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(pre_out_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "out_pre_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    fig, ax = plt.subplots(best_rows, best_cols, figsize=(20, 15))
    for i in range(best_rows):
        for j in range(best_cols):
            sns.heatmap(delta_out_norm_reshaped[i * best_cols + j], cmap="coolwarm", annot=False, square=False, ax=ax[i, j])
            ax[i, j].set_title(f"Layer {i * best_cols + j}")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "out_delta_norm_heatmap.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(pre_q_norm_cls_token, label="pre_q", color="red", marker="o")
    ax.plot(delta_q_norm_cls_token, label="delta_q", color="red", marker="o", linestyle="--")
    ax.plot(pre_k_norm_cls_token, label="pre_k", color="blue", marker="o")
    ax.plot(delta_k_norm_cls_token, label="delta_k", color="blue", marker="o", linestyle="--")
    ax.plot(pre_v_norm_cls_token, label="pre_v", color="green", marker="o")
    ax.plot(delta_v_norm_cls_token, label="delta_v", color="green", marker="o", linestyle="--")
    ax.plot(pre_out_norm_cls_token, label="pre_out", color="orange", marker="o")
    ax.plot(delta_out_norm_cls_token, label="delta_out", color="orange", marker="o", linestyle="--")
    ax.legend()
    ax.grid()
    ax.set_ylim(0, 600)
    ax.set_title("Feature Norms of CLS Token")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Norm")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "feature",
        f"{args.eval_dataset}",
        f"lamb_{args.lamb}",
        "cls_token_norm.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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
    if args.finetuning_type == "lora":
        for key in keys:
            pre_q.append(state_dict[key + ".q_proj.Pre.weight"])
            pre_k.append(state_dict[key + ".k_proj.Pre.weight"])
            pre_v.append(state_dict[key + ".v_proj.Pre.weight"])
            pre_out.append(state_dict[key + ".out_proj.Pre.weight"])
            delta_q.append(state_dict[key + ".q_proj.Delta.B"] @ state_dict[key + ".q_proj.Delta.A"])
            delta_k.append(state_dict[key + ".k_proj.Delta.B"] @ state_dict[key + ".k_proj.Delta.A"])
            delta_v.append(state_dict[key + ".v_proj.Delta.B"] @ state_dict[key + ".v_proj.Delta.A"])
            delta_out.append(state_dict[key + ".out_proj.Delta.B"] @ state_dict[key + ".out_proj.Delta.A"])
    else:
        for key in keys:
            pre_q.append(state_dict[key + ".q_proj.Pre.weight"])
            pre_k.append(state_dict[key + ".k_proj.Pre.weight"])
            pre_v.append(state_dict[key + ".v_proj.Pre.weight"])
            pre_out.append(state_dict[key + ".out_proj.Pre.weight"])
            delta_q.append(state_dict[key + ".q_proj.Delta.D"])
            delta_k.append(state_dict[key + ".k_proj.Delta.D"])
            delta_v.append(state_dict[key + ".v_proj.Delta.D"])
            delta_out.append(state_dict[key + ".out_proj.Delta.D"])

    pre_q_norm = [
        torch.linalg.matrix_norm(q_layer, ord="fro").to("cpu").numpy()
        for q_layer in pre_q
    ]
    delta_q_norm = [
        torch.linalg.matrix_norm(q_layer, ord="fro").to("cpu").numpy()
        for q_layer in delta_q
    ]

    pre_k_norm = [
        torch.linalg.matrix_norm(k_layer, ord="fro").to("cpu").numpy()
        for k_layer in pre_k
    ]
    delta_k_norm = [
        torch.linalg.matrix_norm(k_layer, ord="fro").to("cpu").numpy()
        for k_layer in delta_k
    ]

    pre_v_norm = [
        torch.linalg.matrix_norm(v_layer, ord="fro").to("cpu").numpy()
        for v_layer in pre_v
    ]
    delta_v_norm = [
        torch.linalg.matrix_norm(v_layer, ord="fro").to("cpu").numpy()
        for v_layer in delta_v
    ]

    pre_out_norm = [
        torch.linalg.matrix_norm(out_layer, ord="fro").to("cpu").numpy()
        for out_layer in pre_out
    ]
    delta_out_norm = [
        torch.linalg.matrix_norm(out_layer, ord="fro").to("cpu").numpy()
        for out_layer in delta_out
    ]

    fig, ax = plt.subplots()
    ax.plot(pre_q_norm, label="pre_q", color="red", marker="o")
    ax.plot(delta_q_norm, label="delta_q", color="red", marker="o", linestyle="--")
    ax.plot(pre_k_norm, label="pre_k", color="blue", marker="o")
    ax.plot(delta_k_norm, label="delta_k", color="blue", marker="o", linestyle="--")
    ax.plot(pre_v_norm, label="pre_v", color="green", marker="o")
    ax.plot(delta_v_norm, label="delta_v", color="green", marker="o", linestyle="--")
    ax.plot(pre_out_norm, label="pre_out", color="orange", marker="o")
    ax.plot(delta_out_norm, label="delta_out", color="orange", marker="o", linestyle="--")
    ax.legend()
    ax.grid()
    ax.set_title("Weight Norms")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Norm")
    filename = os.path.join(
        args.fig_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        f"arithmetic_on_{args.pretrained}",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.eval_datasets}",
        "weight_norm",
        f"weight_norm_lamb_{args.lamb}.jpg"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    args: Args = Args().from_args()
    lamb = [
        round(0.1 * i, 2)
        for i in range(21)
    ]
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
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"task_vector_for_{args.eval_datasets}.pt"
    ))

    for lm in lamb:
        print("=" * 100)
        print(f"Processing lambda: {lm}")
        print("=" * 100)
        args.lamb = lm

        image_encoder: ImageEncoder = task_vector.apply_to(pretrained_encoder, args.lamb)
        inspect_weights(image_encoder, args)
        for dataset_name in args.eval_datasets:
            print(f"Processing dataset: {dataset_name}")
            args.eval_dataset = dataset_name
            dataset = get_dataset(args.eval_dataset, image_encoder.val_preprocess, args.dataset_root, batch_size=args.batch_size, num_workers=args.num_workers)
            features = get_inner_features(image_encoder, dataset, args)
            inspect_features(features, args, image_encoder.model.visual.transformer.layers)

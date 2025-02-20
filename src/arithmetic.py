import os
import random
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import torch

from args import Args
from eval import evaluate
from modeling import ImageEncoder
from task_vectors import TaskVector


def eval_task_vectors(pretrained_encoder: ImageEncoder, task_vector: TaskVector, args: Args) -> Dict[str, Dict[str, float]]:
    """Evaluate the task vectors on the pretrained model."""
    print("=" * 100)
    print(f"Evaluating {args.pretrained_to_transfer} Task Vectors on {args.pretrained} pretrained model")
    print("=" * 100)
    print(f"Using Device: {args.device}")

    info = {}

    for coef in args.lamb:
        print("-" * 100)
        print(f"Evaluating with lambda = {coef}")
        print("-" * 100)

        args.result = os.path.join(
            args.result_root,
            args.model_architecture,
            args.pretrained_to_transfer,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            f"arithmetic_on_{args.pretrained}",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"{args.eval_datasets}",
            f"lambda_{coef}.json"
        )
        args.fig = os.path.join(
            args.fig_root,
            args.model_architecture,
            args.pretrained_to_transfer,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            f"arithmetic_on_{args.pretrained}",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"{args.eval_datasets}",
            f"lambda_{coef}.jpg"
        )

        image_encoder = task_vector.apply_to(pretrained_encoder, coef)

        info[f"{coef}"] = evaluate(image_encoder, args)
        print(f"Average accuracy: {info[f'{coef}']['AVG.']:.2%}")

    return info


def plot_coef_vs_average_accuracy(info: Dict[str, Dict[str, float]], args: Args) -> None:
    """Plot the coefficient vs average accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(info.keys(), [info[coef]['AVG.'] for coef in info.keys()])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Average Accuracy")
    ax.set_title(f"Coefficient vs Average Accuracy for {args.pretrained_to_transfer} on {args.pretrained}")
    ax.grid(True)
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
        f"coef_vs_average_accuracy.jpg"
    )
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    args: Args = Args().from_args()
    assert args.eval_datasets is not None, "Evaluation datasets must be specified"
    if args.finetuning_type != "lora":
        args.rank = 0
        args.alpha = 0

    SEED = args.seed
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    args.lamb = (
        [round(0.1 * i, 2) for i in range(0, 21)]
        if args.lamb is None else [args.lamb]
    )

    # Load pretrained model
    pretrained_encoder = ImageEncoder(args, keep_lang=False)
    pretrained_encoder.save(
        os.path.join(
            args.model_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"zeroshot_rank_{args.rank}.pt"
        )
    )
    state_dict = pretrained_encoder.state_dict()
    for key in state_dict.keys():
        if "Delta.D" in key or "Delta.b" in key or "Delta.A" in key or "Delta.B" in key:
            state_dict[key] = torch.zeros_like(state_dict[key])
    pretrained_encoder.load_state_dict(state_dict, strict=False)

    # Load Task Vector
    finetuned_encoders = [
        ImageEncoder.load(os.path.join(
            args.model_root,
            args.model_architecture,
            args.pretrained_to_transfer,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"finetuned_image_encoder_on_{dataset_name}.pt"
        ))
        for dataset_name in args.eval_datasets
    ]
    task_vector = sum([
        TaskVector(
            pretrained_checkpoint=pretrained_encoder,
            finetuned_checkpoint=finetuned_encoder
        )
        for finetuned_encoder in finetuned_encoders
    ])
    task_vector.save_vector(os.path.join(
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

    info = eval_task_vectors(pretrained_encoder, task_vector, args)
    plot_coef_vs_average_accuracy(info, args)

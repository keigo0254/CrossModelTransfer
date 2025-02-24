import os
import random

import numpy as np
import torch

from arithmetic import eval_task_vectors, plot_coef_vs_average_accuracy
from args import Args
from modeling import ImageEncoder
from task_vectors import TaskVector


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
    base_pretrained_encoder = ImageEncoder(args, keep_lang=False)
    base_pretrained_encoder.save(
        os.path.join(
            args.model_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"zeroshot_rank_{args.rank}.pt"
        )
    )
    state_dict = base_pretrained_encoder.state_dict()
    for key in state_dict.keys():
        if "Delta" in key:
            state_dict[key] = torch.zeros_like(state_dict[key])
    base_pretrained_encoder.load_state_dict(state_dict, strict=False)

    # Load Task Vector
    pretrained_encoder_path = os.path.join(
        args.model_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"zeroshot_rank_{args.rank}.pt"
    )
    pretrained_encoder = ImageEncoder.load(pretrained_encoder_path)

    model_vector = TaskVector(
        pretrained_checkpoint=pretrained_encoder,
        finetuned_checkpoint=base_pretrained_encoder
    )
    model_vector.save_vector(os.path.join(
        args.model_root,
        args.model_architecture,
        args.pretrained_to_transfer,
        args.finetuning_type,
        f"model_vector_from_{args.pretrained_to_transfer}_to_{args.pretrained}_rank_{args.rank}.pt"
    ))

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

    info = eval_task_vectors(base_pretrained_encoder, task_vector + model_vector, args)
    plot_coef_vs_average_accuracy(info, args)


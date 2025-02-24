import json
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

from args import Args
from modeling import ImageEncoder
from task_vectors import TaskVector


if __name__ == "__main__":
    args: Args = Args().from_args()
    train_steps = [
        200 * i
        for i in range(32)
    ]
    task_vector_list: List[TaskVector] = [
        TaskVector.load_vector(os.path.join(
            args.model_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
        "orthogonal_finetune",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"{args.train_datasets}",
        f"{args.dataset_type}",
            f"orthogonal_finetune_on_{args.train_dataset}_step_{train_step}_model_vector_{args.model_vector}.pt"
        ))
        for train_step in train_steps
    ]
    final_task_vector = TaskVector.load_vector(os.path.join(
            args.model_root,
            args.model_architecture,
            args.pretrained,
            args.finetuning_type,
            f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
            f"rank_{args.rank}_alpha_{args.alpha}",
            "orthogonal_finetune",
            f"bs_{args.batch_size}_seed_{args.seed}",
            f"{args.train_datasets}",
            f"{args.dataset_type}",
            f"orthogonal_finetuned_task_vector_on_{args.train_datasets}_for_epochs_{args.epochs}_model_vector_{args.model_vector}.pt"
    ))
    final_U_dict = {
        "q": [],
        "k": [],
        "v": [],
        "out": [],
    }
    for key, value in final_task_vector.items():
        if "q_proj.Delta.U" in key:
            final_U_dict["q"].append(value)
        elif "k_proj.Delta.U" in key:
            final_U_dict["k"].append(value)
        elif "v_proj.Delta.U" in key:
            final_U_dict["v"].append(value)
        elif "out_proj.Delta.U" in key:
            final_U_dict["out"].append(value)            

    for task_vector in task_vector_list:
        U_dict = {
            "q": [],
            "k": [],
            "v": [],
            "out": [],
        }
        for key, value in task_vector.items():
            if "q_proj.Delta.U" in key:
                U_dict["q"].append(value)
            elif "k_proj.Delta.U" in key:
                U_dict["k"].append(value)
            elif "v_proj.Delta.U" in key:
                U_dict["v"].append(value)
            elif "out_proj.Delta.U" in key:
                U_dict["out"].append(value)

        for key, value in U_dict.items():
            norm_list =[
                torch.linalg.matrix_norm(v.T @ v, ord="fro")
                for v in value
            ]
            final_norm_list =[
                torch.linalg.matrix_norm(v.T @ v, ord="fro")
                for v in final_U_dict[key]
            ]
            print(f"{key}: norm = {np.mean(norm_list)}, final_norm = {np.mean(final_norm_list)}")

            # 重ねて描画
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(norm_list, label="norm")
            ax.plot(final_norm_list, label="final_norm")
            ax.legend()
            filename = os.path.join(
                args.fig_root,
                args.model_architecture,
                args.pretrained,
                args.finetuning_type,
                f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
                f"rank_{args.rank}_alpha_{args.alpha}",
                "orthogonal_finetune",
                f"bs_{args.batch_size}_seed_{args.seed}",
                f"{args.train_datasets}",
                f"{args.dataset_type}",
                f"U_norm_of_orthogonal_finetune_on_{args.train_datasets}_for_epochs_{args.epochs}_model_vector_{args.model_vector}.png"
            )
            plt.savefig(filename)
            plt.close()

            
        

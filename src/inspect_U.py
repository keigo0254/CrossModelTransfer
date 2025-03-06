import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from args import Args
from modeling import ImageEncoder
from orthogonal_finetune import OrthLoss
from task_vectors import TaskVector


if __name__ == "__main__":
    final_image_encoder = ImageEncoder.load(
        "model/ViT-B-32/laion400m_e32/linear/lr_1e-05_wd_0.1_ls_0.0/rank_0_alpha_0/orthogonal_finetune/bs_128_seed_2025/['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']/cycle/randomize_False_lamb_1.0/orthogonal_finetuned_image_encoder_on_['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']_for_epochs_100_model_vector_False.pt"
    )
    final_U_dict = {
        "q": [],
        "k": [],
        "v": [],
        "out": [],
    }
    state_dict = final_image_encoder.state_dict()
    for key, value in state_dict.items():
        if "q_proj.Delta.U" in key:
            final_U_dict["q"].append(value)
        elif "k_proj.Delta.U" in key:
            final_U_dict["k"].append(value)
        elif "v_proj.Delta.U" in key:
            final_U_dict["v"].append(value)
        elif "out_proj.Delta.U" in key:
            final_U_dict["out"].append(value)
    breakpoint()

    orth_loss = OrthLoss()
    loss = orth_loss(final_image_encoder)
    print(loss)
    breakpoint()

    # for i, task_vector in enumerate(task_vector_list):
    #     U_dict = {
    #         "q": [],
    #         "k": [],
    #         "v": [],
    #         "out": [],
    #     }
    #     D_dict = {
    #         "q": [],
    #         "k": [],
    #         "v": [],
    #         "out": [],
    #     }
    #     for key, value in task_vector.vector.items():
    #         if "q_proj.Delta.U" in key:
    #             U_dict["q"].append(value)
    #             if args.finetuning_type == "lora":
    #                 D_dict["q"].append(task_vector.vector["q_proj.Delta.B"] @ task_vector.vector["q_proj.Delta.A"])
    #             else:
    #                 D_dict["q"].append(task_vector.vector["q_proj.Delta.D"])
    #         elif "k_proj.Delta.U" in key:
    #             U_dict["k"].append(value)
    #             if args.finetuning_type == "lora":
    #                 D_dict["k"].append(task_vector.vector["k_proj.Delta.B"] @ task_vector.vector["k_proj.Delta.A"])
    #             else:
    #                 D_dict["k"].append(task_vector.vector["k_proj.Delta.D"])
    #         elif "v_proj.Delta.U" in key:
    #             U_dict["v"].append(value)
    #             if args.finetuning_type == "lora":
    #                 D_dict["v"].append(task_vector.vector["v_proj.Delta.B"] @ task_vector.vector["v_proj.Delta.A"])
    #             else:
    #                 D_dict["v"].append(task_vector.vector["v_proj.Delta.D"])
    #         elif "out_proj.Delta.U" in key:
    #             U_dict["out"].append(value)
    #             if args.finetuning_type == "lora":
    #                 D_dict["out"].append(task_vector.vector["out_proj.Delta.B"] @ task_vector.vector["out_proj.Delta.A"])
    #             else:
    #                 D_dict["out"].append(task_vector.vector["out_proj.Delta.D"])

    #     for key, value in U_dict.items():
    #         norm_list = [
    #             torch.linalg.matrix_norm(v.T @ v, ord="fro")
    #             for v in value
    #         ]
    #         final_norm_list = [
    #             torch.linalg.matrix_norm(v.T @ v, ord="fro")
    #             for v in final_U_dict[key]
    #         ]
    #         print(f"{key}: norm = {np.mean(norm_list)}, final_norm = {np.mean(final_norm_list)}")

    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.plot(norm_list, label="norm")
    #         ax.plot(final_norm_list, label="final_norm")
    #         ax.set_xlabel("lambda")
    #         ax.set_ylabel("frobenius norm")
    #         ax.legend()
    #         filename = os.path.join(
    #             args.fig_root,
    #             args.model_architecture,
    #             args.pretrained,
    #             args.finetuning_type,
    #             f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
    #             f"rank_{args.rank}_alpha_{args.alpha}",
    #             "orthogonal_finetune",
    #             f"bs_{args.batch_size}_seed_{args.seed}",
    #             f"{args.train_datasets}",
    #             f"{args.dataset_type}",
    #             f"U_norm_of_orthogonal_finetune_on_{args.train_datasets}_for_epochs_{args.epochs}_model_vector_{args.model_vector}_step_{train_steps[i]}.png"
    #         )
    #         plt.savefig(filename)
    #         plt.close()

    #         cosine_similarity_list = [
    #             F.cosine_similarity(u, v, dim=0)
    #             for u, v in zip(U_dict[key], final_U_dict[key])
    #         ]
    #         print(f"{key}: cosine_similarity = {np.mean(cosine_similarity_list)}")
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.plot(cosine_similarity_list, label="cosine_similarity")
    #         ax.set_xlabel("lambda")
    #         ax.set_ylabel("cosine_similarity")
    #         ax.legend()
    #         filename = os.path.join(
    #             args.fig_root,
    #             args.model_architecture,
    #             args.pretrained,
    #             args.finetuning_type,
    #             f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
    #             f"rank_{args.rank}_alpha_{args.alpha}",
    #             "orthogonal_finetune",
    #             f"bs_{args.batch_size}_seed_{args.seed}",
    #             f"{args.train_datasets}",
    #             f"{args.dataset_type}",
    #             f"cosine_similarity_of_orthogonal_finetune_on_{args.train_datasets}_for_epochs_{args.epochs}_model_vector_{args.model_vector}_step_{train_steps[i]}.png"
    #         )
    #         plt.savefig(filename)
    #         plt.close()

    #         UDU_list = [
    #             u.T @ d @ u
    #             for u, d in zip(U_dict[key], D_dict[key])
    #         ]
    #         D_list = [
    #             d for d in D_dict[key]
    #         ]
    #         cosine_similarity_list = [
    #             F.cosine_similarity(u.flatten(), v.flatten(), dim=0)
    #             for u, v in zip(UDU_list, D_list)
    #         ]
    #         print(f"{key}: cosine_similarity of UDU and D = {np.mean(cosine_similarity_list)}")
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.plot(cosine_similarity_list, label="cosine_similarity")
    #         ax.set_xlabel("lambda")
    #         ax.set_ylabel("cosine_similarity")
    #         ax.legend()
    #         filename = os.path.join(
    #             args.fig_root,
    #             args.model_architecture,
    #             args.pretrained,
    #             args.finetuning_type,
    #             f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
    #             f"rank_{args.rank}_alpha_{args.alpha}",
    #             "orthogonal_finetune",
    #             f"bs_{args.batch_size}_seed_{args.seed}",
    #             f"{args.train_datasets}",
    #             f"{args.dataset_type}",
    #             f"cosine_similarity_of_UDU_and_D_of_orthogonal_finetune_on_{args.train_datasets}_for_epochs_{args.epochs}_model_vector_{args.model_vector}_step_{train_steps[i]}.png"
    #         )
    #         plt.savefig(filename)
    #         plt.close()

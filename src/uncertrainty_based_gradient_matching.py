import copy
import os
import shutil
from typing import Dict

import torch
import torch.nn as nn
import open_clip

from args import Args
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from modeling import ImageEncoder as CustomImageEncoder
from task_vectors import TaskVector
import utils


class ImageEncoder(nn.Module):
    def __init__(self, args: Args, keep_lang = False) -> None:
        super().__init__()

        print(f"Loading {args.pretrained} pre-trained weights.")
        self.model, self.train_preprocess, self.val_preprocess = \
            open_clip.create_model_and_transforms(
            model_name=args.model_architecture,
            pretrained=args.pretrained,
            cache_dir=args.cache_root
        )

        self.cache_dir = args.cache_root

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)


@torch.no_grad()
def folding_weights(
    target_model: CustomImageEncoder, args: Args
) -> ImageEncoder:
    new_model = ImageEncoder(args=args, keep_lang=False)
    new_model_state_dict: Dict[str, torch.Tensor] = new_model.state_dict()
    target_model_state_dict: Dict[str, torch.Tensor] = target_model.state_dict()

    num_layers = target_model.model.visual.transformer.layers
    attns = [
        f"model.visual.transformer.resblocks.{i}.attn"
        for i in range(num_layers)
    ]

    for attn in attns:
        if args.finetuning_type == "lora":
            key = f"{attn}.q_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            A = target_model_state_dict[key + ".A"]
            B = target_model_state_dict[key + ".B"]
            U = target_model_state_dict[key + ".U"]
            Q = W + U @ B @ A @ U.T * args.alpha / args.rank

            key = f"{attn}.k_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            A = target_model_state_dict[key + ".A"]
            B = target_model_state_dict[key + ".B"]
            U = target_model_state_dict[key + ".U"]
            K = W + U @ B @ A @ U.T * args.alpha / args.rank

            key = f"{attn}.v_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            A = target_model_state_dict[key + ".A"]
            B = target_model_state_dict[key + ".B"]
            U = target_model_state_dict[key + ".U"]
            V = W + U @ B @ A @ U.T * args.alpha / args.rank

            in_proj_weight = torch.cat([Q, K, V], dim=0)
            new_model_state_dict[f"{attn}.in_proj_weight"] = in_proj_weight

            key = f"{attn}.out_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            A = target_model_state_dict[key + ".A"]
            B = target_model_state_dict[key + ".B"]
            U = target_model_state_dict[key + ".U"]
            O = W + U @ B @ A @ U.T * args.alpha / args.rank

            new_model_state_dict[f"{attn}.out_proj.weight"] = O
        else:
            key = f"{attn}.q_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            D = target_model_state_dict[key + ".D"]
            U = target_model_state_dict[key + ".U"]
            Q = W + U @ D @ U.T
            if key + ".b" in target_model_state_dict:
                Q_b = target_model_state_dict[key.replace("Delta", "Pre.bias")] + target_model_state_dict[key + ".b"]

            key = f"{attn}.k_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            D = target_model_state_dict[key + ".D"]
            U = target_model_state_dict[key + ".U"]
            K = W + U @ D @ U.T
            if key + ".b" in target_model_state_dict:
                K_b = target_model_state_dict[key.replace("Delta", "Pre.bias")] + target_model_state_dict[key + ".b"]

            key = f"{attn}.v_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            D = target_model_state_dict[key + ".D"]
            U = target_model_state_dict[key + ".U"]
            V = W + U @ D @ U.T
            if key + ".b" in target_model_state_dict:
                V_b = target_model_state_dict[key.replace("Delta", "Pre.bias")] + target_model_state_dict[key + ".b"]

            in_proj_weight = torch.cat([Q, K, V], dim=0)
            in_proj_bias = torch.cat([Q_b, K_b, V_b], dim=0) if key + ".b" in target_model_state_dict else None
            new_model_state_dict[f"{attn}.in_proj_weight"] = in_proj_weight
            if in_proj_bias is not None:
                new_model_state_dict[f"{attn}.in_proj_bias"] = in_proj_bias

            key = f"{attn}.out_proj.Delta"
            W = target_model_state_dict[key.replace("Delta", "Pre")]
            D = target_model_state_dict[key + ".D"]
            U = target_model_state_dict[key + ".U"]
            O = W + U @ D @ U.T
            if key + ".b" in target_model_state_dict:
                O_b = target_model_state_dict[key.replace("Delta", "Pre.bias")] + target_model_state_dict[key + ".b"]

            new_model_state_dict[f"{attn}.out_proj.weight"] = O
            if key + ".b" in target_model_state_dict:
                new_model_state_dict[f"{attn}.out_proj.bias"] = O_b

    new_model.load_state_dict(new_model_state_dict)
    return new_model


# def get_hessian():


if __name__ == "__main__":
    args = Args().from_args()

    finetuned_image_encoder = CustomImageEncoder.load(
        args.model_root,
        args.model_architecture,
        args.pretrained,
        args.finetuning_type,
        f"lr_{args.lr}_wd_{args.wd}_ls_{args.ls}",
        f"rank_{args.rank}_alpha_{args.alpha}",
        "finetune",
        f"bs_{args.batch_size}_seed_{args.seed}",
        f"finetuned_image_encoder_on_{args.train_dataset}.pt"
    )

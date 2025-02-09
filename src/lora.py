from argparse import Namespace
import math
import os

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import ImageEncoder, MultiheadAttention, CustomLinear
import utils
from utils import get_submodules


class Scaling(nn.Module):
    def __init__(self, scale: float):
        super(Scaling, self).__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LoRALinear(CustomLinear):
    def __init__(
        self, in_features: int, out_features: int, bias_flag: bool = True,
        r: int = 4, lora_alpha: int = 8, linear: nn.Linear = None
    ):
        super().__init__(in_features, out_features, bias_flag, linear)
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias_flag
        self.r = r
        self.lora_alpha = lora_alpha
        self.W = nn.Linear(in_features, out_features, bias=bias_flag)
        self.W.weight.requires_grad = False
        if self.bias_flag:
            self.W.bias.requires_grad = False
        if linear is not None:
            self.W.weight.data = linear.weight.data.clone()
            if self.bias_flag:
                self.W.bias.data = linear.bias.data.clone()
        if hasattr(self, "D"):
            delattr(self, "D")
        self.D = nn.Sequential(
            nn.Linear(in_features, r, bias=False),
            nn.Linear(r, out_features, bias=False),
            Scaling(lora_alpha / r)
        )
        self.init_lora()
        self.D[0].weight.requires_grad = True
        self.D[1].weight.requires_grad = True

    def init_lora(self):
        nn.init.kaiming_uniform_(self.D[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.D[1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = self.W(x)
        BAx = self.D(x)
        return Wx + BAx


class LoRAImageEncoder(ImageEncoder):
    def __init__(self, args, keep_lang=False):
        super().__init__(args, keep_lang)

        for key, module in self.named_modules():
            if isinstance(module, CustomLinear):
                parent, target_name, target = get_submodules(self, key)
                setattr(
                    parent, target_name,
                    LoRALinear(
                        module.in_features, module.out_features,
                        bias_flag=module.bias_flag,
                        r=args.r, lora_alpha=args.lora_alpha, linear=module.W
                    )
                )

    def freeze(self):
        for key, param in self.named_parameters():
            if "D.0.weight" in key or "D.1.weight" in key:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    @torch.no_grad()
    def reset_params(self):
        for key, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.D[0].weight.copy_(torch.zeros_like(module.D[0].weight))
                module.D[1].weight.copy_(torch.zeros_like(module.D[1].weight))


if __name__ == "__main__":
    from args import parse_arguments

    args = parse_arguments()
    args.model_architecture = "ViT-B-16"
    args.r = 4
    args.lora_alpha = 8
    model = LoRAImageEncoder(args, keep_lang=False)
    model.freeze()
    # print(model.state_dict().keys())
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # x = torch.randn(1, 3, 224, 224)
    # model(x)
    state_dict = model.state_dict()
    for key, param in model.model.named_parameters():
        if "D" in key:
            state_dict[key] = torch.zeros_like(param)
    model.load_state_dict(state_dict)

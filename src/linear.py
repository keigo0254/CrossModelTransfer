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


class DeltaLinear(CustomLinear):
    def __init__(self, in_features: int, out_features: int, bias_flag: bool = True, linear: CustomLinear = None):
        super().__init__(in_features, out_features, bias_flag, linear)
        if hasattr(self, "D"):
            delattr(self, "D")
        self.D = nn.Linear(in_features, out_features, bias=False)
        self.init_parameters()

    def init_parameters(self):
        nn.init.zeros_(self.D.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = self.W(x)
        Dx = self.D(x)
        return Wx + Dx


class LinearImageEncoder(ImageEncoder):
    def __init__(self, args, keep_lang=False):
        super().__init__(args, keep_lang)

        for key, module in self.named_modules():
            if isinstance(module, CustomLinear):
                parent, target_name, target = get_submodules(self, key)
                setattr(
                    parent, target_name,
                    DeltaLinear(
                        module.in_features, module.out_features,
                        bias_flag=module.bias_flag, linear=module.W
                    )
                )

    def freeze(self):
        for key, param in self.named_parameters():
            if "D.weight" in key:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)


if __name__ == "__main__":
    from args import parse_arguments

    args = parse_arguments()
    model = LinearImageEncoder(args, keep_lang=False)
    model.freeze()
    print(model)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    x = torch.randn(1, 3, 224, 224)
    model(x)

import math
import os
from typing import Dict, List

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import Args
import utils
from utils import get_submodules


class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_flag: bool = True, linear: nn.Linear = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias_flag
        self.W = nn.Linear(in_features, out_features, bias=bias_flag)
        self.D = nn.Linear(in_features, out_features, bias=bias_flag)
        self.init_parameters()
        self.W.weight.requires_grad = False
        if self.bias_flag:
            self.W.bias.requires_grad = False
        if linear is not None:
            self.W.weight.data = linear.weight.data.clone()
            if self.bias_flag:
                self.W.bias.data = linear.bias.data.clone()

    def init_parameters(self):
        nn.init.zeros_(self.D.weight)
        nn.init.zeros_(self.D.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = self.W(x)
        Dx = self.D(x)
        return Wx + Dx


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mha: nn.MultiheadAttention = None,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_proj = CustomLinear(self.embed_dim, self.embed_dim, bias_flag=True)
        self.k_proj = CustomLinear(self.embed_dim, self.embed_dim, bias_flag=True)
        self.v_proj = CustomLinear(self.embed_dim, self.embed_dim, bias_flag=True)
        self.o_proj = CustomLinear(self.embed_dim, self.embed_dim, bias_flag=True)

        if mha is not None:
            self.load_existing_layer(mha)

    def load_existing_layer(self, mha: nn.MultiheadAttention) -> None:
        self.q_proj.W.weight.data = mha.in_proj_weight.chunk(3, dim=0)[0]
        self.k_proj.W.weight.data = mha.in_proj_weight.chunk(3, dim=0)[1]
        self.v_proj.W.weight.data = mha.in_proj_weight.chunk(3, dim=0)[2]
        self.k_proj.W.bias.data = mha.in_proj_bias.chunk(3, dim=0)[1]
        self.q_proj.W.bias.data = mha.in_proj_bias.chunk(3, dim=0)[0]
        self.v_proj.W.bias.data = mha.in_proj_bias.chunk(3, dim=0)[2]
        self.o_proj.W.weight.data = mha.out_proj.weight
        self.o_proj.W.bias.data = mha.out_proj.bias
        self.dropout = mha.dropout

    def in_projection(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        return q, k, v

    def multi_head_attention_forward(
        self, query: torch.Tensor, key: torch.Tensor,
        value: torch.Tensor, embed_dim: int,
        num_heads: int, dropout_p: float, training: bool,
        need_weights: bool = True, attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // num_heads

        q, k, v = self.in_projection(query, key, value)
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if not training:
            dropout_p = 0.0

        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask,
                q_scaled,
                k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(
                q_scaled,
                k.transpose(-2, -1)
            )
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(
                attn_output_weights,
                p=dropout_p
            )

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(
                tgt_len * bsz, embed_dim
            )
        )
        attn_output = self.o_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len
        )
        attn_output_weights = attn_output_weights.mean(dim=1)

        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        need_weights: bool = True, attn_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_output_weights = self.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            self.dropout, self.training,
            need_weights, attn_mask
        )
        return attn_output, attn_output_weights


class ImageEncoder(nn.Module):
    def __init__(self, args: Args, keep_lang: bool = False):
        super().__init__()

        print(f"Loading {args.model_architecture} of pre-trained {args.pretrained} weights.")

        self.model, self.train_preprocess, self.val_preprocess = (
            open_clip.create_model_and_transforms(
                args.model_architecture,
                pretrained=args.pretrained,
                cache_dir=args.cache_root
            )
        )

        self.args = args

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

        for key, module in self.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                parent, target_name, target = get_submodules(
                    self, key
                )
                setattr(
                    parent, target_name,
                    MultiheadAttention(
                        module.embed_dim,
                        module.num_heads,
                        mha=module
                    )
                )

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename) -> 'ImageEncoder':
        print(f"Loading image encoder from {filename}")
        return utils.torch_load(filename)

    @classmethod
    def load_from_state_dict(cls, state_dict: Dict[str, torch.Tensor], args: Args = None):
        if args is None:
            args: Args = Args.from_args()

        cls.model, cls.train_preprocess, cls.val_preprocess = (
            open_clip.create_model_and_transforms(
                args.model_architecture,
                pretrained=args.pretrained,
                cache_dir=args.cache_root
            )
        )
        cls.model.load_state_dict(state_dict)
        return cls


class ClassificationHead(nn.Linear):
    def __init__(self, normalize: bool, weights: torch.Tensor, biases: torch.Tensor = None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs: torch.Tensor):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder: ImageEncoder = image_encoder
        self.classification_head: ClassificationHead = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(nn.Module):
    def __init__(self, image_encoder: ImageEncoder, classification_heads: Dict[str, ClassificationHead]):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = classification_heads
        for dataset_name, classification_head in self.classification_heads.items():
            self.add_module(dataset_name, classification_head)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for dataset_name in self.classification_heads.keys():
            self.classification_heads[dataset_name].weight.requires_grad_(False)
            self.classification_heads[dataset_name].bias.requires_grad_(False)

    def forward_id(self, inputs, dataset_name):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[dataset_name](features)
        return outputs

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = {}
        for dataset_name in self.classification_heads.keys():
            outputs[dataset_name] = self.classification_heads[dataset_name](features)

        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


if __name__ == '__main__':
    from eval import evaluate, eval_multihead_classifier
    from heads import get_classification_head


    args: Args = Args.from_args()
    args.eval_datasets = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN",
        "CIFAR10", "CIFAR100", "ImageNet", "STL10"
    ]

    image_encoder = ImageEncoder(args, keep_lang=False)
    # evaluate(image_encoder, args)

    classification_heads = {}
    for dataset in args.eval_datasets:
        classification_head = get_classification_head(args, dataset)
        classification_heads[dataset] = classification_head

    multi_head_image_classifier = MultiHeadImageClassifier(image_encoder, classification_heads)
    eval_multihead_classifier(multi_head_image_classifier, args)

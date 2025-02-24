import math
import os
from typing import Dict, List

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import Args
from delta import Linear, LoRALayer
import utils
from utils import get_submodules


class MultiheadAttention(nn.Module):
    """Customed MultiheadAttention with LoRA or Linear layers"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        enable_lora: list = ['q', 'k', 'v', 'o'],
        r: int = 0,
        lora_alpha: int = 1,
        dropout: float = 0.0,
        original_mha: nn.MultiheadAttention = None,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.enable_lora = enable_lora
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = dropout
        if original_mha._qkv_same_embed_dim:
            q_weight = original_mha.in_proj_weight.chunk(3)[0]
            k_weight = original_mha.in_proj_weight.chunk(3)[1]
            v_weight = original_mha.in_proj_weight.chunk(3)[2]
        else:
            q_weight = original_mha.q_proj_weight
            k_weight = original_mha.k_proj_weight
            v_weight = original_mha.v_proj_weight
        q_bias = original_mha.in_proj_bias[:embed_dim]
        k_bias = original_mha.in_proj_bias[embed_dim:2*embed_dim]
        v_bias = original_mha.in_proj_bias[2*embed_dim:]
        out_weight = original_mha.out_proj.weight
        out_bias = original_mha.out_proj.bias

        if "q" in enable_lora:
            self.q_proj = Linear(embed_dim, embed_dim, r=r, alpha=lora_alpha, weight=q_weight, bias=q_bias)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.q_proj.weight = q_weight
            if q_bias is not None:
                self.q_proj.bias = q_bias
        if "k" in enable_lora:
            self.k_proj = Linear(embed_dim, embed_dim, r=r, alpha=lora_alpha, weight=k_weight, bias=k_bias)
        else:
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.k_proj.weight = k_weight
            if k_bias is not None:
                self.k_proj.bias = k_bias
        if "v" in enable_lora:
            self.v_proj = Linear(embed_dim, embed_dim, r=r, alpha=lora_alpha, weight=v_weight, bias=v_bias)
        else:
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.v_proj.weight = v_weight
            if v_bias is not None:
                self.v_proj.bias = v_bias
        if "o" in enable_lora:
            self.out_proj = Linear(embed_dim, embed_dim, r=r, alpha=lora_alpha, weight=out_weight, bias=out_bias)
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.out_proj.weight = out_weight
            if out_bias is not None:
                self.out_proj.bias = out_bias

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
        attn_output = self.out_proj(attn_output)
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
    """Image encoder with LoRA or Linear layers for MultiheadAttention"""
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
                        r=args.rank,
                        lora_alpha=args.alpha,
                        dropout=args.dropout,
                        original_mha=module
                    )
                )

    def freeze_pretrained_weight(self):
        for name, param in self.named_parameters():
            if "Delta.D" in name or "Delta.A" in name or "Delta.B" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def freeze_except_U(self):
        for name, param in self.named_parameters():
            if "Delta.U" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def freeze_pretrained_weight_and_unfreeze_Delta(self):
        for name, param in self.named_parameters():
            if "Delta" in name and "U" not in name:
                param.requires_grad_(True)
            elif ("attn" in name and "proj" in name) or "U" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def randomize_U(self):
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                module.randomize()
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

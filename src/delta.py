import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: int = 1,
        bias: bool = False,
    ):
        """LoRA or Linear layer with additional weight"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.bias = bias

        if r > 0:
            self.A = nn.Parameter(torch.empty(r, in_features))
            self.B = nn.Parameter(torch.empty(out_features, r))
            self.reset_parameters()
        else:
            self.D = nn.Parameter(torch.zeros(in_features, out_features))
            nn.init.zeros_(self.D)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
                nn.init.zeros_(self.bias)

        self.U = nn.Parameter(torch.eye(in_features))

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            nn.init.zeros_(self.D)
            if self.bias:
                nn.init.zeros_(self.bias)

    def zero_parameters(self):
        if self.r > 0:
            nn.init.zeros_(self.A)
            nn.init.zeros_(self.B)
        else:
            nn.init.zeros_(self.D)
            if self.bias:
                nn.init.zeros_(self.bias)

    @torch.no_grad()
    def qr(self):
        Q, R = torch.linalg.qr(self.U)
        self.U.copy_(Q)

    def forward(self, inputs: torch.Tensor):
        if self.r > 0:
            return F.linear(inputs, self.U.T @ self.B @ self.A @ self.U) * self.alpha / self.r
        else:
            if self.bias:
                return F.linear(inputs, self.U.T @ self.D @ self.U, self.bias)
            else:
                return F.linear(inputs, self.U.T @ self.D @ self.U)

    def __repr__(self):
        return f'LoRALayer(in_features={self.in_features}, out_features={self.out_features}, r={self.r}, alpha={self.alpha})'


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: int = 1,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """Wrapper class with LoRA or Linear layer for MultiheadAttention"""
        self.r = r
        self.alpha = alpha

        super().__init__(in_features, out_features, bias=True if bias is not None else False)
        if weight is not None:
            self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)

        self.Delta = LoRALayer(in_features, out_features, r, alpha)
        self.Delta.reset_parameters()

    def forward(self, inputs: torch.Tensor):
        return super().forward(inputs) + self.Delta(inputs)


if __name__ == "__main__":
    bias_linear = Linear(10, 10, bias=nn.Parameter(torch.zeros(10)))
    for key, param in bias_linear.named_parameters():
        print(key, param.shape)
    print()
    non_bias_linear = Linear(10, 10)
    for key, param in non_bias_linear.named_parameters():
        print(key, param.shape)
    print()
    lora_linear = Linear(10, 10, r=4, alpha=8)
    for key, param in lora_linear.named_parameters():
        print(key, param.shape)
    print()
    x = torch.randn(10)
    print(bias_linear(x))
    print(non_bias_linear(x))
    print(lora_linear(x))

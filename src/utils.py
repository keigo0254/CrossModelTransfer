import os
import pickle
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def get_submodules(
    module: nn.Module, key: str
) -> tuple[nn.Module, str, nn.Module]:
    parent = module.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = module.get_submodule(key)
    return parent, target_name, target


def assign_learning_rate(param_group: Dict, new_lr: float):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_length: int, step: int):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    warmup_length: int,
    steps: int,
) -> callable:
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path):
    import torch
    version = tuple(map(int, torch.__version__.split(".")[:2]))
    if version >= (2, 6):
        import torch.serialization
        from modeling import ClassificationHead, ImageEncoder
        torch.serialization.add_safe_globals([ClassificationHead, ImageEncoder])
        return torch.load(save_path, weights_only=False)
    else:
        return torch.load(save_path)


def get_logits(inputs: torch.Tensor, classifier) -> torch.Tensor:
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

import copy
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from utils import torch_load, torch_save


class TaskVector:
    """Create a task vector between a pretrained and finetuned model"""
    def __init__(
        self,
        pretrained_checkpoint: Optional[nn.Module] = None,
        finetuned_checkpoint: Optional[nn.Module] = None,
        vector: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = pretrained_checkpoint.state_dict()
                finetuned_state_dict = finetuned_checkpoint.state_dict()
                self.vector = {}
                # Exclude integer parameters
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other: 'TaskVector') -> 'TaskVector':
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other: Union[None, int, 'TaskVector']) -> 'TaskVector':
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self) -> 'TaskVector':
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint: nn.Module, scaling_coef: float = 1.0) -> nn.Module:
        with torch.no_grad():
            pretrained_model = copy.deepcopy(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f'Warning: key {key} is present in the pretrained state dict '
                        f'but not in the task vector'
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

    def save_vector(self, path: str) -> None:
        print(f'Saving task vector to {path}')
        torch_save(self.vector, path)

    def load_vector(self, path: str) -> None:
        print(f'Loading task vector from {path}')
        self.vector = torch_load(path)


if __name__ == "__main__":
    from args import Args
    from eval import evaluate
    from modeling import ImageEncoder

    args: Args = Args().from_args()
    args.pretrained = "openai"
    image_encoder_1 = ImageEncoder(args, keep_lang=False)
    args.pretrained = "laion400m_e32"
    image_encoder_2 = ImageEncoder(args, keep_lang=False)
    task_vector = TaskVector(image_encoder_1, image_encoder_2)
    new_image_encoder = task_vector.apply_to(image_encoder_1, 1.0)
    print(new_image_encoder)
    args.pretrained = "laion400m_e32"
    args.eval_datasets = ["Cars"]
    evaluate(new_image_encoder, args)

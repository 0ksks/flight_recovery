from dataclasses import dataclass
import torch as th


@dataclass
class Experience:
    state: th.Tensor
    action: th.Tensor
    reward: th.Tensor
    done: th.Tensor

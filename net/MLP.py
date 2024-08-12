import torch.nn as nn
from torch import device as torch_device
from collections import OrderedDict
from typing import Literal
from . import activation_mapping_func


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        arch: list[int],
        hidden_activation: Literal["relu", "sigmoid"] = "relu",
        in_activation: Literal["relu", "sigmoid"] = "relu",
        out_activation: Literal["relu", "sigmoid", "identity"] = "sigmoid",
        device: torch_device = torch_device("cpu"),
    ):
        """
        simplest MLP
        """
        super(MLP, self).__init__()
        layers_dict = OrderedDict()
        layers_dict["in_proj"] = nn.Linear(input_size, arch[0])
        layers_dict["in_act"] = activation_mapping_func(in_activation)
        arch_len = len(arch)
        if arch_len > 1:
            hidden_in_dim = arch[:-1]
            hidden_out_dim = arch[1:]
            for idx, (hidden_in, hidden_out) in enumerate(
                zip(hidden_in_dim, hidden_out_dim)
            ):
                layers_dict[f"hidden_{idx}"] = nn.Linear(hidden_in, hidden_out)
                layers_dict[f"hidden_act_{idx}_{idx+1}"] = activation_mapping_func(
                    hidden_activation
                )
            layers_dict.pop(f"hidden_act_{arch_len-2}_{arch_len-1}")
        else:
            layers_dict["hidden_0"] = nn.Linear(arch[0], arch[-1])
        layers_dict[f"hidden_out"] = activation_mapping_func(hidden_activation)
        layers_dict["out_proj"] = nn.Linear(arch[-1], output_size)
        layers_dict["out_act"] = activation_mapping_func(out_activation)
        self.layers_dict = layers_dict
        self.device = device
        self.layers = None

    def register_layers(self) -> None:
        for key, layer in self.layers_dict.items():
            self.layers_dict[key] = layer.to(self.device)
        self.layers = nn.Sequential(self.layers_dict)

    def forward(self, x):
        assert (
            self.layers is not None
        ), "Please call register_layers() before calling MLP()"
        return self.layers(x)

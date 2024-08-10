import torch.nn as nn
from collections import OrderedDict
from . import activation_mapping_func


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        arch: list[int],
        hidden_activation: str = "relu",
        in_activation: str = "relu",
        out_activation: str = "sigmoid",
    ):
        """
        simplest MLP
        """
        super(MLP, self).__init__()
        layers = OrderedDict()
        layers["in_proj"] = nn.Linear(input_size, arch[0])
        layers["in_act"] = activation_mapping_func(in_activation)
        if len(arch) > 1:
            hidden_in_dim = arch[:-1]
            hidden_out_dim = arch[1:]
            for idx, (hidden_in, hidden_out) in enumerate(
                zip(hidden_in_dim, hidden_out_dim)
            ):
                layers[f"hidden_{idx}"] = nn.Linear(hidden_in, hidden_out)
                layers[f"act_{idx}_{idx+1}"] = activation_mapping_func(
                    hidden_activation
                )
            layers.pop(f"act_{len(arch) - 2}_{len(arch) - 1}")
        else:
            layers["hidden_0"] = nn.Linear(arch[0], arch[-1])
        layers["out_act"] = activation_mapping_func(out_activation)
        layers["out_proj"] = nn.Linear(arch[-1], output_size)
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)

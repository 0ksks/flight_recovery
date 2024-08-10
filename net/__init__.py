import torch.nn as nn


def activation_mapping_func(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()

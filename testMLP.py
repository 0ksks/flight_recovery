import torch as th
from tensordict.nn.distributions import NormalParamExtractor
from net.MLP import MLP

model = MLP(4, 2, [3, 4, 4, 4], out_activation="identity")
# normalizer = NormalParamExtractor()
# model.layers_dict["normalizer"] = normalizer
model.register_layers()
print(model)

input_tensor = th.randn(5, 4)
output = model(input_tensor)
print(output)

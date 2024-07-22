import torch

from net.GraphEncoder import GAT
from utils.DataIO import get_data, transform_problems_into_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = GAT(
        num_of_layers=2, num_heads_per_layer=[1, 1], num_features_per_layer=[1, 2, 3]
    )
    model = model.to(DEVICE)
    problems, solutions = get_data("data")
    node_features, edge_index = transform_problems_into_tensor(problems[0], DEVICE)
    print(node_features, edge_index)
    output = model((node_features, edge_index))
    print(output)

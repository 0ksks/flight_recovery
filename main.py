import torch

from net.GraphEncoder import GAT
from utils.DataIO import get_data, transform_problems_into_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = GAT(
        num_of_layers=2, num_heads_per_layer=[1, 1], num_features_per_layer=[1, 1, 1]
    )
    model = model.to(DEVICE)
    problems, solutions = get_data("data")
    edge_index, node_features = transform_problems_into_tensor(problems[0], DEVICE)
    model((edge_index, node_features))

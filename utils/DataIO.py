import pickle
import os

import numpy as np
import torch


def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        return data


def save_data(data_path, data):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
        print(f"Data saved in {data_path}")


def transform_problems_into_tensor(datum: tuple[np.ndarray], device):
    edge_index = torch.tensor(datum[0]).to(dtype=torch.long, device=device)
    node_features = torch.tensor(datum[1]).to(dtype=torch.float, device=device)
    return node_features, edge_index


def get_problems(data_path):
    problems = load_data(os.path.join(data_path, "problems.pkl"))
    return problems


def get_solutions(data_path):
    solutions = load_data(os.path.join(data_path, "solutions.pkl"))
    return solutions


def get_data(data_path):
    problems = get_problems(data_path)
    solutions = get_solutions(data_path)
    return problems, solutions

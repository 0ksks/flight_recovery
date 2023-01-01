import pickle
import networkx as nx
import numpy as np


def get_weight(edge: tuple[int, int], node_weight: np.ndarray, shortest):
    if shortest:
        return (1 - node_weight[edge[0]]) + (1 - node_weight[edge[1]])
    return node_weight[edge[0]] + node_weight[edge[1]]


def solve(edge_index: np.ndarray, node_attr: np.ndarray, shortest=True):
    G = nx.DiGraph()

    weight_fun = lambda u, v, d: get_weight((u, v), node_attr, shortest)

    # 添加带权重的节点
    for i, weight in enumerate(node_attr):
        G.add_node(i, weight=weight)

    G.add_edges_from(edge_index.T)

    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra(G, weight=weight_fun))
    return all_pairs_shortest_paths, G


class Problem:

    def __init__(self, num_of_points: int):
        upper_tri = np.triu(
            np.random.randint(0, 2, size=(num_of_points, num_of_points)), k=1
        )
        self.edge_index = np.array(np.where(upper_tri == 1))

        edge_attr = np.random.rand(upper_tri.sum())
        self.edge_attr = edge_attr

        node_attr = np.random.rand(num_of_points)
        self.node_attr = node_attr

    @property
    def weighted_edge_shortest_path(self):
        return solve(self.edge_index, self.node_attr)

    def to_data(self):
        """
        :return:
            edge_index[[begin],[end]]
            node_attr[[weight]]
            shortest_path{begin:
                                ({end:path_length},
                                {end:path})}
        """
        return (
            self.edge_index,
            np.expand_dims(self.node_attr, axis=1),
        ), self.weighted_edge_shortest_path[0]


if __name__ == "__main__":
    # problems = []
    # solutions = []
    # for _ in range(10):
    #     problem, solution = Problem(np.random.randint(10, 21)).to_data()
    #     problems.append(problem)
    #     solutions.append(solution)
    # pickle.dump(problems, open("problems.pkl", "wb"))
    # pickle.dump(solutions, open("solutions.pkl", "wb"))
    print(Problem(np.random.randint(10, 21)).to_data())

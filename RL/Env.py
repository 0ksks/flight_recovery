from typing import Union

import numpy as np
from numpy import array

from RL import AdjList


class DAGmap:
    def __init__(self, adj_list: AdjList, reward=1):
        self.adj_list = adj_list
        self.n_nodes = len(self.adj_list)
        self.reward = reward
        self.visited = np.zeros(self.n_nodes).astype(bool)

    def avail_next(self, node_idx: int) -> np.ndarray:
        all_next = self.adj_list[node_idx]
        return all_next[~self.visited[all_next]]

    def actor(self, node: int) -> tuple[list[int], list[float]]: ...

    def critic(self, node: int, action: int) -> float: ...


class Aeroplane:
    def __init__(self, start_node: int, end_node: int, reward=1, info=None):
        self.start_node = start_node
        self.end_node = end_node
        self.current_node = self.start_node
        self.reward = reward
        self.info = info

    def step(self, node: int):
        self.current_node = node
        reward = self.reward
        terminated = self.current_node == self.end_node
        return reward, terminated

    def actor(self, node: int, mask: list[bool]) -> list[float]: ...

    def critic(self, node: int, action: int) -> float: ...


class MultiAeroplane:
    def __init__(
        self,
        start_node_array: list[int],
        end_node_array: list[int],
        reward_array: Union[list, int] = 1,
        info_array=None,
    ):
        if not isinstance(reward_array, list):
            reward_array = [reward_array] * len(start_node_array)
        if info_array is None:
            info_array = [None] * len(start_node_array)
        self.aeroplanes = [
            Aeroplane(start_node, end_node, reward, info)
            for start_node, end_node, reward, info in zip(
                start_node_array, end_node_array, reward_array, info_array
            )
        ]


if __name__ == "__main__":
    dag = DAGmap(
        {
            0: array([1, 2]).astype(int),
            1: array([2, 3]).astype(int),
            2: array([4]).astype(int),
            3: array([4]).astype(int),
        }
    )
    dag.visited[1] = True
    dag.visited[2] = True
    print(dag.avail_next(0))

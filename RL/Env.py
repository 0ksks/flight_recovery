from typing import Union
import numpy as np
from numpy import array
from RL import AdjList
from MultiDecision import MultiAct
import torch as th


class DAGmap:
    def __init__(self, adj_list: AdjList, reward=1):
        self.adj_list = adj_list
        self.n_nodes = len(self.adj_list)
        self.reward = reward
        self.visited = np.zeros(self.n_nodes).astype(bool)
        self.actor_network = None
        self.critic_network = None

    def visit(self, node):
        self.visited[node] = True

    def register_network(self, actor_network=None, critic_network=None):
        self.actor_network = actor_network
        self.critic_network = critic_network

    def avail_next(self, node_idx: int) -> np.ndarray:
        all_next = self.adj_list[node_idx]
        return all_next[~self.visited[all_next]]

    def actor(self, node: int, mask: list[bool]) -> tuple[list[int], list[float]]:
        assert (
            self.actor_network is not None
        ), "actor network is not initialized, use `register_network()`"
        return self.actor_network(node, mask)

    def critic(self, node: int, action: int) -> float:
        assert (
            self.critic_network is not None
        ), "critic network is not initialized, use `register_network()`"
        return self.critic_network(node, action)

    def step(self, action_array: list[int]):
        """
        :param action_array: (1 * action_indices)
        :return: reward_array, terminated_array, next_step_array
        """
        self.visit(action_array)
        terminated_array = [
            self.avail_next(action).size == 0 for action in action_array
        ]
        reward_array = [
            0 if terminated else self.reward for terminated in terminated_array
        ]
        return reward_array, terminated_array, action_array

    def get_state(self):
        """visited node list[bool]"""
        return self.visited


class Aeroplane:
    def __init__(self, start_node: int, end_node: int, reward=1, info=None):
        self.start_node = start_node
        self.end_node = end_node
        self.current_node = self.start_node
        self.reward = reward
        self.info = info
        self.actor_network = None
        self.critic_network = None

    def register_network(self, actor_network=None, critic_network=None):
        self.actor_network = actor_network
        self.critic_network = critic_network

    def actor(self, node: int, mask: list[bool]) -> list[float]:
        assert (
            self.actor_network is not None
        ), "actor network is not initialized, use `register_network()`"
        return self.actor_network(node, mask)

    def critic(self, node: int, action: int) -> float:
        assert (
            self.critic_network is not None
        ), "critic network is not initialized, use `register_network()`"
        return self.critic_network(node, action)

    def step(self, node: int):
        self.current_node = node
        reward = self.reward
        terminated = self.current_node == self.end_node
        return reward, terminated

    def get_state(self):
        """current node int"""
        return self.current_node


class MultiAeroplane:
    def __init__(
        self,
        start_node_array: Union[list[int], th.Tensor],
        end_node_array: Union[list[int], th.Tensor],
        reward_array: Union[list, th.Tensor, int] = 1,
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

    def actor(
        self,
        node_array: Union[list[int], th.Tensor],
        mask_array: Union[list[bool], th.Tensor],
    ) -> th.Tensor:
        """
        return the index of the best action in each row
        :return: max_index: (1 * agent_num)
        """
        prob = []
        for aeroplane, node, mask in zip(self.aeroplanes, node_array, mask_array):
            prob.append(aeroplane.actor(node, mask))
        prob = th.tensor(prob)
        decisions = MultiAct.decide(prob)
        return decisions

    def critic(
        self,
        node_array: Union[list[int], th.Tensor],
        action_list: Union[list[int], th.Tensor],
    ):
        q_value = []
        for aeroplane, node, action in zip(self.aeroplanes, node_array, action_list):
            q_value.append(aeroplane.critic(node, action))
        q_value = th.tensor(q_value)
        return q_value


if __name__ == "__main__":
    dag = DAGmap(
        {
            0: array([1, 2]).astype(int),
            1: array([2, 3]).astype(int),
            2: array([4]).astype(int),
            3: array([4]).astype(int),
            4: array([]).astype(int),
        }
    )
    # dag.visited[1] = True
    # dag.visited[2] = True
    dag.step([0])
    res = dag.step([1, 2])
    print(res)
    res = dag.step([4])
    print(res)
    print(dag.avail_next(0))

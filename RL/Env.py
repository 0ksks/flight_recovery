from typing import Union
from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch as th

from .MultiDecision import MultiAct


class DAGmap(gym.Env):
    def __init__(
        self,
        adj_list: dict[int, Union[list[int], np.ndarray, th.Tensor]],
        reward=1,
        device=th.device("cpu"),
    ):
        self.adj_list = adj_list
        self.n_states = len(self.adj_list)
        self.reward = reward
        self.__visited = np.zeros(self.n_states).astype(bool)
        self.actor_network = None
        self.critic_network = None
        self.device = device

    @staticmethod
    def adj_list_2_edge_idx(
        adj_list: dict[int, Union[list[int], np.ndarray, th.Tensor]]
    ) -> list[list[int]]:
        edge_idx = []
        for start in adj_list:
            for end in adj_list[start]:
                edge_idx.append([start, end])
        edge_idx = np.array(edge_idx).T.tolist()
        return edge_idx

    @staticmethod
    def edge_idx_2_adj_list(
        edge_idx: list[list[int]],
    ) -> dict[int, Union[list[int], np.ndarray, th.Tensor]]:
        adj_list = defaultdict(list)
        cols = len(edge_idx[0])
        for i in range(cols):
            adj_list[edge_idx[0][i]].append(edge_idx[1][i])
        adj_list = dict(adj_list)
        for i in range(cols):
            if edge_idx[1][i] not in adj_list.keys():
                adj_list[edge_idx[1][i]] = []
        return adj_list

    @staticmethod
    def np_array_adj_list(adj_list: dict[int, list[int]]):
        adj_list_array: dict[int, np.ndarray] = {}
        for k, v in adj_list.items():
            adj_list_array[k] = np.array(v).astype(int)
        return adj_list_array

    def visit(self, state) -> bool:
        updated = not self.__visited[np.argwhere(state == 1)].all()
        self.__visited[state] = True
        return updated

    def register_network(self, actor_network=None, critic_network=None):
        self.actor_network = actor_network
        self.critic_network = critic_network

    def avail_next(self, state_idx: int) -> np.ndarray:
        all_next = self.adj_list[state_idx]
        return all_next[~self.__visited[all_next]]

    def actor(
        self, state: Union[int, th.Tensor], mask: Union[list[bool], th.Tensor]
    ) -> Union[list[float], th.Tensor]:
        assert (
            self.actor_network is not None
        ), "actor network is not initialized, use `self.register_network(actor_network, critic_network)`"
        return self.actor_network(state, mask)

    def critic(
        self, state: Union[int, th.Tensor], action: Union[int, th.Tensor]
    ) -> Union[float, th.Tensor]:
        assert (
            self.critic_network is not None
        ), "critic network is not initialized, use `self.register_network(actor_network, critic_network)`"
        return self.critic_network(state, action)

    def step(self, action_array: Union[list[int], np.ndarray, th.Tensor]):
        """
        :return: observation, reward, terminated, truncated, info
        """
        visited_array = self.get_state().copy()
        updated = self.visit(action_array)
        terminated_array = [
            self.avail_next(action_idx).size == 0
            for action_idx, _ in enumerate(action_array)
        ]
        if updated:
            reward_array = [
                self.reward if action == 1 and not visited else 0
                for action, visited in zip(action_array, visited_array)
            ]
        else:
            reward_array = [0 for _ in action_array]
        return action_array, sum(reward_array), terminated_array, False, {}

    def reset(self):
        self.__visited = np.zeros(self.n_states).astype(bool)
        return self.__visited, {}

    def _reset(self, tensordict):
        return self.reset()

    def get_state(self):
        """visited state list[bool]"""
        return self.__visited


class Aeroplane:
    def __init__(self, start_state: int, end_state: int, reward=1, info=None):
        self.start_state = start_state
        self.end_state = end_state
        self.current_state = self.start_state
        self.reward = reward
        self.info = info
        self.actor_network = None
        self.critic_network = None

    def register_network(self, actor_network=None, critic_network=None):
        self.actor_network = actor_network
        self.critic_network = critic_network

    def actor(
        self, aeroplane_state: int, dag_state: list[bool], mask: list[bool]
    ) -> list[float]:
        assert (
            self.actor_network is not None
        ), "actor network is not initialized, use `self.register_network(actor_network, critic_network)`"
        return self.actor_network(aeroplane_state, dag_state, mask)

    def critic(self, aeroplane_state: int, dag_state: list[bool], action: int) -> float:
        assert (
            self.critic_network is not None
        ), "critic network is not initialized, use `self.register_network(actor_network, critic_network)`"
        return self.critic_network(aeroplane_state, dag_state, action)

    def step(
        self,
        aeroplane_action: int,
        dag_state: list[bool],
    ):
        self.current_state = aeroplane_action
        reward = self.reward if self.current_state == self.end_state else 0
        terminated = self.current_state == self.end_state
        return reward, terminated

    def get_state(self):
        """current state int"""
        if not isinstance(self.current_state, int):
            return int(self.current_state)
        return self.current_state


class MultiAeroplane:
    def __init__(
        self,
        start_state_array: Union[list[int], th.Tensor],
        end_state_array: Union[list[int], th.Tensor],
        reward_array: Union[list, th.Tensor, int] = 1,
        info_array=None,
    ):
        if not isinstance(reward_array, list):
            reward_array = [reward_array] * len(start_state_array)
        if info_array is None:
            info_array = [None] * len(start_state_array)
        self.aeroplanes = [
            Aeroplane(start_state, end_state, reward, info)
            for start_state, end_state, reward, info in zip(
                start_state_array, end_state_array, reward_array, info_array
            )
        ]
        self.start_state = start_state_array
        self.end_state = end_state_array
        self.aeroplane_num = len(self.aeroplanes)

    def actor(
        self,
        aeroplane_state_array: Union[list[int], th.Tensor],
        dag_state: Union[list[bool], th.Tensor],
        mask_array: Union[list[list[bool]], th.Tensor],
    ) -> th.Tensor:
        """
        return the index of the best action in each row
        :param aeroplane_state_array: current state for each aeroplane (1 * n_aeroplanes)
        :param dag_state: visited nodes (1 * n_nodes)
        :param mask_array: mask for each aeroplane (n_aeroplanes * n_aeroplane_states)
        :return: max_index: (1 * n_aeroplanes)
        """
        prob = []
        for aeroplane, aeroplane_state, dag_state, mask in zip(
            self.aeroplanes, aeroplane_state_array, dag_state, mask_array
        ):
            prob.append(aeroplane.actor(aeroplane_state, dag_state, mask))
        prob = th.tensor(np.array(prob))
        decisions = MultiAct.decide(prob)
        return decisions

    def critic(
        self,
        aeroplane_state_array: Union[list[int], th.Tensor],
        dag_state: Union[list[bool], th.Tensor],
        action_array: Union[list[int], th.Tensor],
    ) -> th.Tensor:
        q_value = []
        for aeroplane, aeroplane_state, dag_state, action in zip(
            self.aeroplanes, aeroplane_state_array, dag_state, action_array
        ):
            q_value.append(aeroplane.critic(aeroplane_state, dag_state, action))
        q_value = th.tensor(np.array(q_value))
        return q_value

    def step(
        self,
        aeroplane_action_array: Union[list[int], th.Tensor],
        dag_state: Union[list[bool], th.Tensor, None],
    ):
        """
        :param aeroplane_action_array:
        :param dag_state:
        :return: [[reward, terminated]*n_aeroplanes]
        """
        # reward_array = []
        # terminated_array = []
        # for aeroplane, aeroplane_action in zip(self.aeroplanes, aeroplane_action_array):
        #     reward, terminated = aeroplane.step(aeroplane_action, dag_state)
        #     reward_array.append(reward)
        #     terminated_array.append(terminated)
        # result = [reward_array, terminated_array]
        # tensor_result = th.tensor(result).transpose(0, 1)
        # return tensor_result
        result = list(
            map(
                lambda tup: tup[0].step(tup[1], dag_state),
                zip(self.aeroplanes, aeroplane_action_array),
            )
        )
        result = th.tensor(result)
        return result

    def get_state(self):
        # states = []
        # for aeroplane in self.aeroplanes:
        #     states.append(aeroplane.get_state())
        # states = th.tensor(states).unsqueeze(0).transpose(0, 1)
        states = list(map(lambda x: x.get_state(), self.aeroplanes))
        states = th.tensor(states).unsqueeze(0).transpose(0, 1)
        return states

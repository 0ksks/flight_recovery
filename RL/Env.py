import random
from typing import Union
import numpy as np
from RL import AdjList
from MultiDecision import MultiAct
import torch as th


class DAGmap:
    def __init__(self, adj_list: AdjList, reward=1):
        self.adj_list = adj_list
        self.n_states = len(self.adj_list)
        self.reward = reward
        self.__visited = np.zeros(self.n_states).astype(bool)
        self.actor_network = None
        self.critic_network = None

    def visit(self, state):
        self.__visited[state] = True

    def register_network(self, actor_network=None, critic_network=None):
        self.actor_network = actor_network
        self.critic_network = critic_network

    def avail_next(self, state_idx: int) -> np.ndarray:
        all_next = self.adj_list[state_idx]
        return all_next[~self.__visited[all_next]]

    def actor(self, state: int, mask: list[bool]) -> list[float]:
        assert (
            self.actor_network is not None
        ), "actor network is not initialized, use `self.register_network(actor_network, critic_network)`"
        return self.actor_network(state, mask)

    def critic(self, state: int, action: int) -> float:
        assert (
            self.critic_network is not None
        ), "critic network is not initialized, use `self.register_network(actor_network, critic_network)`"
        return self.critic_network(state, action)

    def step(self, action_array: list[int]):
        """
        :param action_array: (1 * action_indices)
        :return: reward_array, terminated_array, next_state_array
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
        reward = self.reward
        terminated = self.current_state == self.end_state
        return reward, terminated

    def get_state(self):
        """current state int"""
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
        dag_state: Union[list[bool], th.Tensor],
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


if __name__ == "__main__":
    # dag = DAGmap(
    #     {
    #         0: array([1, 2]).astype(int),
    #         1: array([2, 3]).astype(int),
    #         2: array([4]).astype(int),
    #         3: array([4]).astype(int),
    #         4: array([]).astype(int),
    #     }
    # )
    # # dag.visited[1] = True
    # # dag.visited[2] = True
    # dag.step([0])
    # res = dag.step([1, 2])
    # print(res)
    # res = dag.step([4])
    # print(res)
    # print(dag.avail_next(0))

    # aeroplane_test = Aeroplane(1, 4)
    # print(aeroplane_test.get_state())
    # print(aeroplane_test.step(2, None))
    # print(aeroplane_test.get_state())
    # print(aeroplane_test.step(3, None))
    # print(aeroplane_test.get_state())
    # print(aeroplane_test.step(4, None))
    # print(aeroplane_test.get_state())

    multi_aeroplane_test = MultiAeroplane(
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    )
    # print(multi_aeroplane_test.step([2, 3, 4, 5], None))
    # print(multi_aeroplane_test.step([3, 4, 5, 6], None))
    # print(multi_aeroplane_test.step([5, 6, 7, 8], None))
    mask_array_np = np.random.randint(0, 2, (4, 8))
    mask_array_tensor = th.tensor(mask_array_np).to(dtype=th.bool)

    def test_actor(state, dag, mask):
        return np.random.randint(0, 2, (mask.shape[0],))

    for aeroplane in multi_aeroplane_test.aeroplanes:
        aeroplane.register_network(actor_network=test_actor)

    actions = multi_aeroplane_test.actor(
        [0, 1, 2, 3],
        [True, True, True, True, False, False, False, False],
        mask_array_tensor,
    )
    output = multi_aeroplane_test.step(actions, None)
    print(multi_aeroplane_test.get_state())
    print(output)

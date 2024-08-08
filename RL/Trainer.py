from dataclasses import dataclass
from typing import Union

from Env import DAGmap, MultiAeroplane
import torch as th
import numpy as np


@dataclass
class Experience:
    state: th.Tensor
    action: th.Tensor
    reward: th.Tensor
    done: th.Tensor


class AeroplanesRollout:
    def __init__(self, dag: DAGmap, multiAeroplane: MultiAeroplane):
        self.dag = dag
        self.multiAeroplane = multiAeroplane
        self.__init_start_node()

    def multi_aeroplane_network_register(
        self, actor_network_array, critic_network_array
    ):
        for aeroplane, actor_network, critic_network in zip(
            self.multiAeroplane.aeroplanes, actor_network_array, critic_network_array
        ):
            aeroplane.register_network(
                actor_network=actor_network, critic_network=critic_network
            )

    def __init_start_node(self):
        start_node_array = [
            aeroplane.start_state for aeroplane in self.multiAeroplane.aeroplanes
        ]
        self.dag.visit(start_node_array)

    def rollout_episode(
        self,
        episode_steps: int,
    ) -> list[Experience]:
        """
        :param episode_steps:
        :return: aeroplane episode
        """
        step = 0
        aeroplane_episode = []
        done = False
        while step < episode_steps and not done:
            dag_state = self.dag.get_state().copy()
            aeroplane_state_array_tensor = self.multiAeroplane.get_state().flatten()
            dag_state[aeroplane_state_array_tensor] = True

            # aeroplanes act
            aeroplane_mask_array = []
            for aeroplane in self.multiAeroplane.aeroplanes:
                mask = np.zeros(self.dag.n_states).astype(bool)
                mask[self.dag.avail_next(aeroplane.get_state())] = True
                aeroplane_mask_array.append(mask.tolist())
            aeroplane_action_array_tensor = self.multiAeroplane.actor(
                aeroplane_state_array=aeroplane_state_array_tensor,
                dag_state=th.tensor(dag_state),
                mask_array=th.tensor(aeroplane_mask_array),
            )

            # dag acts
            dag_action = np.zeros(self.dag.n_states).astype(bool)
            dag_action[aeroplane_action_array_tensor] = True

            # get aeroplanes info
            aeroplane_reward_terminated_array = self.multiAeroplane.step(
                aeroplane_action_array_tensor, dag_state
            )
            aeroplane_reward_array_tensor, aeroplane_terminated_array_tensor = (
                aeroplane_reward_terminated_array[:, 0],
                aeroplane_reward_terminated_array[:, 1],
            )

            # store experience
            aeroplane_episode.append(
                Experience(
                    aeroplane_state_array_tensor,
                    aeroplane_action_array_tensor,
                    aeroplane_reward_array_tensor,
                    aeroplane_terminated_array_tensor,
                )
            )

            # update step
            step += 1

            # termination check
            done = self.dag.get_state().all()

        return aeroplane_episode


class DAGRollout:
    def __init__(self, dag: DAGmap, start_node: Union[int, th.Tensor]):
        self.dag = dag
        self.start_node = start_node
        self.current_node = self.start_node

    def dag_network_register(self, actor_network, critic_network):
        self.dag.register_network(
            actor_network=actor_network, critic_network=critic_network
        )

    def rollout_episode(self, episode_steps: int) -> list[Experience]:
        step = 0
        dag_episode = []
        done = False
        while step < episode_steps and not done:
            state = self.current_node
            mask = self.dag.avail_next(state)
            prob = self.dag.actor(th.tensor(state), th.tensor(mask))
            action = int(np.array(prob).argmax())
            reward, done, next_state = self.dag.step([action])
            reward = th.Tensor(reward[0])
            done = th.Tensor(done[0])
            action = th.Tensor(action)
            self.current_node = next_state[0]
            dag_episode.append(Experience(state, action, reward, done))
        return dag_episode


if __name__ == "__main__":
    aeroplane_num = 4
    node_num = 16

    def np_array_adj_list(adj_list: dict[int, list[int]]):
        adj_list_array: dict[int, np.ndarray] = {}
        for k, v in adj_list.items():
            adj_list_array[k] = np.array(v).astype(int)
        return adj_list_array

    graph = {
        0: [4, 5],
        1: [6],
        2: [7],
        3: [8, 9],
        4: [10],
        5: [11],
        6: [12],
        7: [13],
        8: [14],
        9: [15],
        10: [],
        11: [],
        12: [],
        13: [],
        14: [],
        15: [],
    }
    dag = DAGmap(
        np_array_adj_list(graph),
    )
    multi_aeroplane_test = MultiAeroplane(
        [0, 1, 2, 3],
        [12, 13, 14, 15],
    )

    def aeroplane_test_actor(state_, env_state_, mask_):
        mask_ = np.array(mask_)
        prob = np.random.rand(mask_.shape[0])
        prob[~mask_] = 0
        return prob

    rollout_test = AeroplanesRollout(dag, multi_aeroplane_test)
    multi_aeroplane_actor = [aeroplane_test_actor for _ in range(aeroplane_num)]
    multi_aeroplane_critic = [None for _ in range(aeroplane_num)]
    rollout_test.multi_aeroplane_network_register(
        multi_aeroplane_actor, multi_aeroplane_critic
    )
    aeroplane_episode = rollout_test.rollout_episode(
        episode_steps=8,
    )
    print(*aeroplane_episode, sep="\n")

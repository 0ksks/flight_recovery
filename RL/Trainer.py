from typing import Union
from dataclasses import dataclass
from Env import DAGmap, MultiAeroplane
import torch as th
import numpy as np


@dataclass
class Experience:
    state: th.Tensor
    action: th.Tensor
    reward: th.Tensor
    done: th.Tensor


class Rollout:
    def __init__(self, dag: DAGmap, multiAeroplane: MultiAeroplane):
        self.dag = dag
        self.multiAeroplane = multiAeroplane
        self.__init_start_node()

    def dag_network_register(self, actor_network, critic_network):
        self.dag.register_network(
            actor_network=actor_network, critic_network=critic_network
        )

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
        dag_mask: list[bool],
    ) -> tuple[list[Experience], list[Experience]]:
        """
        :param episode_steps:
        :param dag_mask:
        :param aeroplane_mask_array:
        :return: DAG episode, aeroplane episode
        """
        step = 0
        dag_state = self.dag.get_state()

        # aeroplane_state_array = []
        # for aeroplane in self.multiAeroplane.aeroplanes:
        #     aeroplane_state_array.append(aeroplane.current_node)

        # aeroplane_state_array = list(
        #     map(
        #         lambda aeroplane: aeroplane.current_state,
        #         self.multiAeroplane.aeroplanes,
        #     )
        # )

        dag_episode = []
        aeroplane_episode = []
        while step < episode_steps:
            dag_state = self.dag.get_state()
            aeroplane_state_array_tensor = self.multiAeroplane.get_state().flatten()
            dag_state[aeroplane_state_array_tensor] = True
            # dag acts
            dag_action_prob = np.array(self.dag.actor(state=dag_state, mask=dag_mask))
            dag_action = np.where(dag_action_prob == dag_action_prob.max(), 1, 0)
            # aeroplanes act
            aeroplane_mask_array = [
                dag_state,
            ] * self.multiAeroplane.aeroplane_num
            aeroplane_action_array_tensor = self.multiAeroplane.actor(
                aeroplane_state_array=aeroplane_state_array_tensor,
                dag_state=dag_state,
                mask_array=aeroplane_mask_array,
            )
            # get dag info
            dag_reward, dag_terminated, dag_new_state = self.dag.step(dag_action)
            aeroplane_reward_terminated_array = self.multiAeroplane.step(
                aeroplane_action_array_tensor, dag_state
            )
            # get aeroplanes info
            aeroplane_reward_array_tensor, aeroplane_terminated_array_tensor = (
                aeroplane_reward_terminated_array[:, 0],
                aeroplane_reward_terminated_array[:, 1],
            )
            # transform into Tensor
            dag_state_tensor = th.tensor(dag_state)
            dag_action_tensor = th.tensor(dag_action)
            dag_reward_tensor = th.tensor(dag_reward)
            dag_terminated_tensor = th.tensor(dag_terminated)
            # store experience
            dag_episode.append(
                Experience(
                    dag_state_tensor,
                    dag_action_tensor,
                    dag_reward_tensor,
                    dag_terminated_tensor,
                )
            )

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
            if dag_terminated_tensor.all():
                break
        return dag_episode, aeroplane_episode


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
    dag_mask = np.zeros(node_num).astype(bool)
    dag_mask[multi_aeroplane_test.start_state] = True
    dag_mask = dag_mask.tolist()

    def aeroplane_test_actor(state_, env_state_, mask_):
        mask_ = np.array(mask_)
        prob = np.random.rand(mask_.shape[0])
        return prob

    def dag_test_actor(state_, mask_):
        return aeroplane_test_actor(state_, None, mask_)

    for aeroplane in multi_aeroplane_test.aeroplanes:
        aeroplane.register_network(actor_network=aeroplane_test_actor)

    rollout_test = Rollout(dag, multi_aeroplane_test)
    rollout_test.dag_network_register(actor_network=dag_test_actor, critic_network=None)
    multi_aeroplane_actor = [aeroplane_test_actor for _ in range(aeroplane_num)]
    multi_aeroplane_critic = [None for _ in range(aeroplane_num)]
    rollout_test.multi_aeroplane_network_register(
        multi_aeroplane_actor, multi_aeroplane_critic
    )
    dag_episode, aeroplane_episode = rollout_test.rollout_episode(
        dag_mask=dag_mask,
        episode_steps=8,
    )
    print(dag_episode)
    print(aeroplane_episode)

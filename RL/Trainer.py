from typing import Union
from dataclasses import dataclass
from Env import DAGmap, MultiAeroplane
import torch as th


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
            aeroplane.start_node for aeroplane in self.multiAeroplane.aeroplanes
        ]
        self.dag.visit(start_node_array)

    def rollout_episode(
        self,
        episode_steps: int,
        dag_mask: list[bool],
        aeroplanes_mask_array: Union[list[list[bool]], th.Tensor],
    ) -> tuple[list[Experience], list[Experience]]:
        """
        :param episode_steps:
        :param dag_mask:
        :param aeroplanes_mask_array:
        :return: DAG episode, aeroplane episode
        """
        step = 0
        dag_state = self.dag.get_state()

        # aeroplane_state_array = []
        # for aeroplane in self.multiAeroplane.aeroplanes:
        #     aeroplane_state_array.append(aeroplane.current_node)
        aeroplane_state_array = list(
            map(
                lambda aeroplane: aeroplane.current_node, self.multiAeroplane.aeroplanes
            )
        )
        aeroplane_state_array = th.tensor(aeroplane_state_array)
        dag_episode = []
        aeroplane_episode = []
        while step < episode_steps:
            # dag acts
            dag_action, _ = self.dag.actor(state=dag_state, mask=dag_mask)
            # aeroplanes act
            aeroplane_action_array_tensor = self.multiAeroplane.actor(
                aeroplane_state_array=aeroplane_state_array,
                dag_state=dag_state,
                mask_array=aeroplanes_mask_array,
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

            aeroplane_state_array_tensor = th.tensor(aeroplane_state_array)
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
            # update dag and aeroplanes state
            dag_state = self.dag.get_state()
            aeroplane_state_array = self.multiAeroplane.get_state()
            # update step
            step += 1
            # termination check
            if dag_terminated_tensor.all():
                break
        return dag_episode, aeroplane_episode

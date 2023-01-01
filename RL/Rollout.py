from typing import Union
import torch as th
import numpy as np

from .ExperienceBuffer import Experience
from .Env import DAGmap, MultiAeroplane


class AeroplanesRollout:
    def __init__(self, dag: DAGmap, multiAeroplane: MultiAeroplane):
        self.dag = dag
        self.multiAeroplane = multiAeroplane
        self.__init_start_node()

    def register_network(self, actor_network_array):
        for aeroplane, actor_network in zip(
            self.multiAeroplane.aeroplanes, actor_network_array
        ):
            aeroplane.register_network(actor_network=actor_network, critic_network=None)

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
        episode = []
        done = False
        while step < episode_steps and not done:
            dag_state = self.dag.get_state().copy()
            aeroplane_state_array_tensor = self.multiAeroplane.get_state().flatten()
            dag_state[aeroplane_state_array_tensor] = True

            # aeroplanes act
            aeroplane_mask_array = []
            for aeroplane in self.multiAeroplane.aeroplanes:
                mask = np.ones(self.dag.n_states).astype(bool)
                mask[self.dag.avail_next(aeroplane.get_state())] = False
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
            episode.append(
                Experience(
                    aeroplane_state_array_tensor,
                    aeroplane_action_array_tensor,
                    aeroplane_reward_array_tensor,
                    aeroplane_terminated_array_tensor.to(th.bool),
                )
            )

            # update step
            step += 1

            # termination check
            done = self.dag.get_state().all()

        return episode


class DAGRollout:
    def __init__(self, dag: DAGmap, start_node: Union[int, th.Tensor]):
        self.dag = dag
        self.start_node = start_node
        self.current_node = self.start_node
        self.dag.visit(self.start_node)

    def register_network(self, actor_network):
        self.dag.register_network(actor_network=actor_network, critic_network=None)

    def rollout_episode(self, episode_steps: int) -> list[Experience]:
        step = 0
        dag_episode = []
        done = False
        while step < episode_steps and not done:
            state = self.current_node
            mask = np.ones(self.dag.n_states).astype(bool)
            mask[self.dag.avail_next(state)] = False
            prob = self.dag.actor(th.tensor(state), th.tensor(mask))
            action = np.zeros(self.dag.n_states).astype(int)
            action[int(np.array(prob).argmax())] = 1
            reward, done, next_state = self.dag.step(action)
            reward = th.tensor(reward)
            done = self.dag.avail_next(np.where(next_state == 1)[0].item()).size == 0
            action = th.argwhere(th.tensor(action) == 1).flatten().item()
            self.current_node = action
            dag_episode.append(
                Experience(th.tensor(state), th.tensor(action), reward, th.tensor(done))
            )
            step += 1
        return dag_episode

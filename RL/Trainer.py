from Env import DAGmap, MultiAeroplane
from collections import namedtuple

Experience = namedtuple("Experience", ["state", "action", "reward", "done"])


class Rollout:
    def __init__(self, dag: DAGmap, multiAeroplane: MultiAeroplane):
        self.dag = dag
        self.multiAeroplane = multiAeroplane

    def dag_register_actor(self, actor_network):
        self.dag.register_network(actor_network=actor_network)

    def multi_aeroplane_register_actor(self, actor_network_array):
        for aeroplane, actor_network in zip(
            self.multiAeroplane.aeroplanes, actor_network_array
        ):
            aeroplane.register_network(actor_network=actor_network)

    def init_start_node(self):
        start_node_array = [
            aeroplane.start_node for aeroplane in self.multiAeroplane.aeroplanes
        ]
        self.dag.visit(start_node_array)

    def rollout_episode(self, episode_steps: int) -> list[Experience]:
        step = 0
        dag_state = self.dag.get_state()

        while step < episode_steps:
            ...

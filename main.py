from RL.Rollout import AeroplanesRollout, DAGRollout
from RL.Env import MultiAeroplane, DAGmap
import numpy as np

if __name__ == "__main__":
    aeroplane_num = 4
    node_num = 16
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
    dag_map = DAGmap(DAGmap.np_array_adj_list(graph))
    multi_aeroplane = MultiAeroplane(
        [0, 1, 2, 3],
        [12, 13, 14, 15],
    )

    def aeroplane_test_actor(state_, env_state_, mask_):
        mask_ = np.array(mask_)
        prob = np.random.rand(mask_.shape[0])
        prob[mask_] = 0
        return prob

    aeroplanes_rollout = AeroplanesRollout(dag_map, multi_aeroplane)
    multi_aeroplane_actor = [aeroplane_test_actor for _ in range(aeroplane_num)]
    aeroplanes_rollout.register_network(multi_aeroplane_actor)
    aeroplane_episode = aeroplanes_rollout.rollout_episode(
        episode_steps=8,
    )
    print(*aeroplane_episode, sep="\n")

    def dag_test_actor(state_, mask_):
        mask_ = np.array(mask_)
        prob = np.random.rand(mask_.shape[0])
        prob[mask_] = 0
        # print(prob, mask_)
        return prob

    dag_rollout = DAGRollout(dag_map, 0)
    dag_rollout.register_network(dag_test_actor)
    dag_episode = dag_rollout.rollout_episode(episode_steps=8)
    print(*dag_episode, sep="\n")

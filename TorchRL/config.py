import torch as th
from torchrl.envs.utils import ExplorationType

#  网络训练更新次数 =
#    (experience.total_frames/experience.frames_per_batch)
#  * train.num_epochs
#  * (experience.frames_per_batch/experience.sub_batch_size)

#  真正送去网络更新的 batch_size 就是 experience.sub_batch_size ，其他的是强化学习里面的值，
#  不过根据上面几个等式，如果 experience.sub_batch_size 比较大，那 experience.frames_per_batch
#  就要相应地变大一点避免除出来太小，从而 experience.total_frames 也要变大一点

#  网络输入：2^node_num 种， 即 node_num 位的二进制数
#  网络输出：node_num 种， 即 node_num 长的 one_hot

config_dict = {
    "graph": {"node_num": 20},  # 有向无环图的结点数
    "network": {
        "mlp_hidden_arch": [64, 64],  # MLP 隐藏层结构
        "lr": 3e-4,
    },
    "train": {
        "num_epochs": 10,
        "sub_batch_size": 64,
        "max_grad_norm": 1.0,
        "eval_exploration_type": ExplorationType.RANDOM,
    },
    "test": {
        "num_steps": 1_000,  # 测试步数
    },
    "device": {
        "is_fork": th.multiprocessing.get_start_method() == "fork",
        "device": (
            th.device(0)
            if th.cuda.is_available()
            and not (th.multiprocessing.get_start_method() == "fork")
            else th.device("cpu")
        ),
    },
    "experience": {
        "total_frames": 50_000,
        "frames_per_batch": 1_000,
        "sub_batch_size": 64,
        "init_stats_num_iter": 1_000,  # 初始化步数
    },
    "ppo": {
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_eps": 0.2,
        "entropy_eps": 1e-4,
    },
}

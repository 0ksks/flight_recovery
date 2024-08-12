import torch as th
import matplotlib.pyplot as plt
from TorchRL.Envs import DAGenv
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    TransformedEnv,
    StepCounter,
)
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import OneHotCategorical
from collections import defaultdict
from tqdm import tqdm

from data.DataGenerator import Problem
from net.MLP import MLP


class Trainer:
    def __init__(self, config_dict: dict):

        env = self.get_env(
            config_dict["graph"]["node_num"], config_dict["device"]["device"]
        )
        env.transform[0].init_stats(
            num_iter=config_dict["experience"]["init_stats_num_iter"],
            reduce_dim=0,
            cat_dim=0,
        )
        self.env = env
        self.policy_module = self.get_policy_module(
            config_dict["network"]["mlp_hidden_arch"], config_dict["device"]["device"]
        )
        self.value_module = self.get_value_module(
            config_dict["network"]["mlp_hidden_arch"], config_dict["device"]["device"]
        )
        self.collector, self.replay_buffer = self.get_collector_and_replay_buffer(
            config_dict["experience"]["frames_per_batch"],
            config_dict["experience"]["total_frames"],
            config_dict["device"]["device"],
        )
        self.advantage_module, self.loss_module = self.get_advantage_and_loss(
            config_dict["ppo"]["gamma"],
            config_dict["ppo"]["lambda"],
            config_dict["ppo"]["clip_eps"],
            config_dict["ppo"]["entropy_eps"],
        )
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(
            config_dict["network"]["lr"],
            config_dict["experience"]["frames_per_batch"],
            config_dict["experience"]["total_frames"],
        )
        self.logs = None
        self.config_dict = config_dict

    @staticmethod
    def get_env(node_num: int, device: th.device) -> TransformedEnv:
        dag_env = DAGenv.from_problem(
            Problem(node_num), th_dtype=th.float, device=device
        )
        env = TransformedEnv(
            dag_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
            device=device,
        )
        return env

    def get_policy_module(
        self, network_arch: list[int], device: th.device
    ) -> ProbabilisticActor:
        dag_env = self.env.base_env
        actor_net = MLP(
            dag_env.state_size, dag_env.action_size, network_arch, device=device
        )
        actor_net.register_layers()
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["logits"]
        )
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["logits"],
            distribution_class=OneHotCategorical,
            return_log_prob=True,
        )
        return policy_module

    def get_value_module(
        self, network_arch: list[int], device: th.device
    ) -> ValueOperator:
        dag_env = self.env.base_env
        value_net = MLP(
            dag_env.state_size,
            1,
            network_arch,
            out_activation="identity",
            device=device,
        )
        value_net.register_layers()
        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )
        return value_module

    def get_collector_and_replay_buffer(
        self,
        frames_per_batch: int,
        total_frames: int,
        device: th.device,
    ) -> tuple[SyncDataCollector, ReplayBuffer]:
        collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        return collector, replay_buffer

    def get_advantage_and_loss(
        self,
        gamma: float,
        lambda_: float,
        clip_eps: float,
        entropy_eps: float,
    ) -> tuple[GAE, ClipPPOLoss]:
        advantage_module = GAE(
            gamma=gamma,
            lmbda=lambda_,
            value_network=self.value_module,
            average_gae=True,
        )

        loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=clip_eps,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default, but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )
        return advantage_module, loss_module

    def get_optimizer_and_scheduler(
        self, lr: float, frames_per_batch: int, total_frames: int
    ) -> tuple[th.optim.Optimizer, th.optim.lr_scheduler.LRScheduler]:
        optim = th.optim.Adam(self.loss_module.parameters(), lr)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )
        return optim, scheduler

    def train(
        self,
    ):
        self.logs = defaultdict(list)
        pbar = tqdm(total=self.config_dict["experience"]["total_frames"])
        eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(self.collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(self.config_dict["train"]["num_epochs"]):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(
                    self.config_dict["experience"]["frames_per_batch"]
                    // self.config_dict["experience"]["sub_batch_size"]
                ):
                    sub_data = self.replay_buffer.sample(
                        self.config_dict["experience"]["sub_batch_size"]
                    )
                    loss_vals = self.loss_module(
                        sub_data.to(self.config_dict["device"]["device"])
                    )
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory, but it's good practice to keep
                    # your gradient norm bounded
                    th.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(),
                        self.config_dict["train"]["max_grad_norm"],
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = f"average reward={self.logs['reward'][-1]: 4.4f} (init={self.logs['reward'][0]: 4.4f})"
            self.logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {self.logs['step_count'][-1]}"
            self.logs["lr"].append(self.optimizer.param_groups[0]["lr"])
            lr_str = f"lr policy: {self.logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(
                    self.config_dict["train"]["eval_exploration_type"]
                ), th.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = self.env.rollout(
                        self.config_dict["test"]["num_steps"], self.policy_module
                    )
                    print(eval_rollout["next", "reward"])
                    self.logs["eval reward"].append(
                        eval_rollout["next", "reward"].mean().item()
                    )
                    self.logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    self.logs["eval step_count"].append(
                        eval_rollout["step_count"].max().item()
                    )
                    eval_str = (
                        f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {self.logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(
                ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
            )

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            self.scheduler.step()

    def plot_logs(self):
        assert self.logs is not None, "logs is None"
        logs = self.logs
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()

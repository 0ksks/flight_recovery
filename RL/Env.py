from typing import Any, TypeVar, Union

import gymnasium as gym
import torch
import numpy as np

ObsType = TypeVar("ObsType", bound=torch.Tensor)
ActType = TypeVar("ActType", bound=torch.Tensor)
RenderFrame = TypeVar("RenderFrame")
NO_ACTION: ActType = torch.tensor(0)
NO_OBS: ObsType = torch.tensor(0)


class DAGEnv(gym.Env):
    def __init__(self, topology: torch.Tensor) -> None:
        """
        :param topology: edge_index
        """
        self.topology = topology
        self.visited = torch.zeros(topology.shape[1], dtype=torch.bool)

    def step(self, action_: ActType):
        next_state_ = action_
        self.update_visited(action_)
        reward_ = 1
        terminal_ = torch.all(self.visited)
        truncated_ = False
        info_ = {}
        return next_state_, reward_, terminal_, truncated_, info_

    def query_unvisited(self, vertex_) -> torch.Tensor:
        return self.topology[:, (self.topology[0, :] == vertex_) & (~self.visited)][
            1, :
        ]

    def update_visited(self, vertex_) -> None:
        self.visited[self.topology[1, :] == vertex_] = True

    def reset(
        self,
        *,
        seed_: Union[int, None] = None,
        options_: Union[dict[str, Any], None] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.visited = self.topology
        return self.visited, {}

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]: ...


class DAGAgentBase:
    def __init__(self, start_, end_) -> None:
        self.start = start_
        self.end = end_

    def choose_action(self, *args, **kwargs) -> ObsType: ...


class DAGAgentRollout(DAGAgentBase):
    def __init__(self, start_, end_) -> None:
        super().__init__(start_, end_)

    def choose_action(self, state_, env: DAGEnv) -> tuple[ActType, bool]:
        action_space_ = env.query_unvisited(state_)
        if action_space_.numel() == 0:
            action_ = NO_ACTION
            terminal_ = True
            return action_, terminal_
        else:
            action_ = torch.tensor(np.random.choice(action_space_))
            terminal_ = action_ == self.end
            return action_, terminal_


if __name__ == "__main__":
    # try
    from data.DataGenerator import Problem

    problem = Problem(20)
    edge_index = problem.to_data()[0][0]
    dagEnv = DAGEnv(torch.tensor(edge_index))
    dagAgentRollout = DAGAgentRollout(1, 9)

    state = 1
    state_history = [state]
    done = False
    while not done:
        cur_action, agent_terminal = dagAgentRollout.choose_action(state, dagEnv)
        next_state, reward, env_terminal, env_truncated, _ = dagEnv.step(cur_action)
        done = agent_terminal or env_terminal or env_truncated
        state_history.append(next_state)
        state = next_state
    print(edge_index.T)
    print(state_history)

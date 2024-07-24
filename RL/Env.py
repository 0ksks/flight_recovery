from typing import Any, TypeVar, Union

import gymnasium as gym
import torch
import numpy as np

ObsType = TypeVar("ObsType", bound=torch.Tensor)
ActType = TypeVar("ActType", bound=torch.Tensor)
RenderFrame = TypeVar("RenderFrame")


class DAGEnv(gym.Env):
    def __init__(self, topology: torch.Tensor) -> None:
        """
        :param topology: edge_index
        """
        self.topology = topology
        self.visited = torch.zeros(topology.shape[1], dtype=torch.bool)

    def step(self, action: ActType):
        next_state_ = action
        self.update_visited(action)
        reward_ = 1
        terminal_ = self.query_unvisited(next_state_).numel() == 0
        truncated_ = False
        info_ = {}
        return next_state_, reward_, terminal_, truncated_, info_

    def query_unvisited(self, vertex) -> torch.Tensor:
        return self.topology[:, (self.topology[0, :] == vertex) & (~self.visited)][1, :]

    def update_visited(self, vertex) -> None:
        self.visited[self.topology[1, :] == vertex] = True

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[dict[str, Any], None] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.visited = self.topology
        return self.visited, {}

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]: ...


class DAGAgentBase:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def choose_action(self, *args, **kwargs) -> ObsType: ...


class DAGAgentRollout(DAGAgentBase):
    def __init__(self, start, end) -> None:
        super().__init__(start, end)

    def choose_action(self, action_space) -> tuple[ActType, bool]:
        action_ = np.random.choice(action_space)
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
    state_history = [
        state,
    ]
    done = False
    while not done:
        cur_action_space = dagEnv.query_unvisited(state)
        cur_action, agent_terminal = dagAgentRollout.choose_action(cur_action_space)
        next_state, reward, env_terminal, env_truncated, _ = dagEnv.step(cur_action)
        done = agent_terminal or env_terminal or env_truncated
        state_history.append(next_state)
        state = next_state
    print(edge_index.T)
    print(state_history)

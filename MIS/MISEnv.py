"""Independent-set construction with explicit finished-rollout padding."""

from dataclasses import dataclass
import networkx as nx
import torch


@dataclass
class ResetMISState:
    problems: list


@dataclass
class MISState:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None


class MISEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        self.FLAG__use_saved_problems = env_params.get(
            "FLAG__use_saved_problems", False
        )

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        if self.FLAG__use_saved_problems:
            self.problems = self.saved_problems[
                self.saved_index : self.saved_index + batch_size
            ]
            self.saved_index += batch_size
        else:
            self.problems = [
                nx.barabasi_albert_graph(self.problem_size, 4)
                for _ in range(batch_size)
            ]
        self.BATCH_IDX = torch.arange(batch_size)[:, None].expand(-1, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(batch_size, -1)

    def reset(self):
        shape = (self.batch_size, self.pomo_size)
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(*shape, 0, dtype=torch.long)
        self.blocked = torch.zeros(*shape, self.problem_size, dtype=torch.bool)
        self.finished = torch.zeros(shape, dtype=torch.bool)
        self.node_count = torch.zeros(shape)
        self.step_state = MISState(
            self.BATCH_IDX,
            self.POMO_IDX,
            ninf_mask=torch.zeros(*shape, self.problem_size),
            finished=self.finished,
        )
        return ResetMISState(self.problems), None, False

    def pre_step(self):
        return self.step_state, None, False

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected.unsqueeze(-1)), dim=-1
        )
        for b, graph in enumerate(self.problems):
            for p in range(self.pomo_size):
                if not self.finished[b, p]:
                    node = int(selected[b, p])
                    self.blocked[b, p, node] = True
                    for neighbor in graph.neighbors(node):
                        self.blocked[b, p, neighbor] = True
        newly_done = self.blocked.all(-1) & ~self.finished
        self.node_count[newly_done] = self.selected_count
        self.finished |= newly_done
        mask = torch.zeros_like(self.blocked, dtype=torch.float32).masked_fill(
            self.blocked, float("-inf")
        )
        mask[self.finished] = 0
        self.step_state.current_node = selected
        self.step_state.ninf_mask = mask
        self.step_state.finished = self.finished
        done = bool(self.finished.all())
        return self.step_state, self.node_count if done else None, done

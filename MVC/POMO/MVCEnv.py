"""Vertex-cover construction on simple undirected graphs."""

from dataclasses import dataclass
import networkx as nx
import torch


@dataclass
class Reset_State:
    problems: list


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None


class MVCEnv:
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
        self.edges = [
            torch.tensor(list(g.edges()), dtype=torch.long).reshape(-1, 2)
            for g in self.problems
        ]
        self.BATCH_IDX = torch.arange(batch_size)[:, None].expand(-1, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(batch_size, -1)

    def reset(self):
        shape = (self.batch_size, self.pomo_size)
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(*shape, 0, dtype=torch.long)
        self.visited = torch.zeros(*shape, self.problem_size, dtype=torch.bool)
        self.node_count = torch.zeros(shape)
        empty = torch.tensor([len(e) == 0 for e in self.edges], dtype=torch.bool)
        self.finished = empty[:, None].expand(shape).clone()
        mask = torch.zeros(*shape, self.problem_size)
        self.step_state = Step_State(
            self.BATCH_IDX, self.POMO_IDX, ninf_mask=mask, finished=self.finished
        )
        return Reset_State(self.problems), None, bool(self.finished.all())

    def pre_step(self):
        done = bool(self.finished.all())
        return self.step_state, -self.node_count if done else None, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected.unsqueeze(-1)), dim=-1
        )
        self.visited[self.BATCH_IDX, self.POMO_IDX, selected] = True
        for b, edges in enumerate(self.edges):
            covered = (
                self.visited[b, :, edges[:, 0]] | self.visited[b, :, edges[:, 1]]
            ).all(-1)
            newly_done = covered & ~self.finished[b]
            self.node_count[b, newly_done] = self.selected_count
            self.finished[b] |= covered
        mask = torch.zeros_like(self.visited, dtype=torch.float32).masked_fill(
            self.visited, float("-inf")
        )
        mask[self.finished] = 0
        self.step_state.current_node = selected
        self.step_state.ninf_mask = mask
        self.step_state.finished = self.finished
        done = bool(self.finished.all())
        return self.step_state, -self.node_count if done else None, done

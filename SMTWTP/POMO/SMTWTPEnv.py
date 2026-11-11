from dataclasses import dataclass
import torch

#  single machine total weighted tardiness problem


def get_random_problems(batch_size, problem_size):
    due_time_norm = torch.rand(size=(batch_size, problem_size))
    due_time = due_time_norm * (problem_size)
    weights = torch.rand(size=(batch_size, problem_size))
    processing_time = torch.rand(size=(batch_size, problem_size))
    return due_time, weights, processing_time


@dataclass
class Reset_State:
    due_time: torch.Tensor = None
    # shape: (batch, problem)
    weights: torch.Tensor = None
    # shape: (batch, problem)
    processing_time: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class SMTWTPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        self.FLAG__use_saved_problems = env_params.get(
            "FLAG__use_saved_problems", False
        )

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)

        self.due_time = None
        self.weights = None
        self.processing_time = None

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            self.due_time, self.weights, self.processing_time = get_random_problems(
                batch_size, self.problem_size
            )
        else:
            self.due_time = self.saved_due_time[
                self.saved_index : self.saved_index + batch_size
            ]
            self.weights = self.saved_weights[
                self.saved_index : self.saved_index + batch_size
            ]
            self.processing_time = self.saved_processing_time[
                self.saved_index : self.saved_index + batch_size
            ]
            self.saved_index += batch_size

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size
        )

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long
        )
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, self.problem_size)
        )
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return (
            Reset_State(self.due_time, self.weights, self.processing_time),
            reward,
            done,
        )

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = (
            float("-inf")
        )
        # shape: (batch, pomo, node)

        # returning values
        done = self.selected_count == self.problem_size
        if done:
            reward = -self._get_reward()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_reward(self):
        weights = (
            self.weights.unsqueeze(1)
            .expand(self.batch_size, self.pomo_size, self.problem_size)
            .gather(dim=2, index=self.selected_node_list)
        )
        due_time = (
            self.due_time.unsqueeze(1)
            .expand(self.batch_size, self.pomo_size, self.problem_size)
            .gather(dim=2, index=self.selected_node_list)
        )
        processing_time = (
            self.processing_time.unsqueeze(1)
            .expand(self.batch_size, self.pomo_size, self.problem_size)
            .gather(dim=2, index=self.selected_node_list)
        )
        presum_time = processing_time.cumsum(dim=-1)
        diff = presum_time - due_time
        diff[diff < 0] = 0
        return (weights * diff).sum(dim=-1)

# Includes adaptations from POMO, Copyright (c) 2021 Yeong-Dae Kwon.
# Upstream MIT license: licenses/POMO-MIT.txt.
import torch


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems

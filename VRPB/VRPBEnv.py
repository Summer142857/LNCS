# Adapted from the Routing-MVMoE environment lineage.
# Copyright (c) 2023 Jianan; upstream MIT license: licenses/Routing-MVMoE-MIT.txt.
"""Mixed delivery/pickup routing with route-defined initial delivery loads."""

from dataclasses import dataclass
import numpy as np
import torch
from CVRP.CVRProblemDef import augment_xy_data_by_8_fold
from lncs.problems import VRP_CAPACITIES


@dataclass
class Reset_State:
    depot_xy: torch.Tensor
    node_xy: torch.Tensor
    node_demand: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    START_NODE: torch.Tensor
    selected_count: int = 0
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    load: torch.Tensor = None


class VRPBEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params["problem_size"]
        self.requested_pomo_size = env_params["pomo_size"]
        self.pomo_size = self.requested_pomo_size
        self.device = torch.device(env_params.get("device", "cpu"))
        self.backhaul_ratio = 0.2

    def get_random_problems(self, batch_size, problem_size):
        depot = torch.rand(batch_size, 1, 2, device=self.device)
        nodes = torch.rand(batch_size, problem_size, 2, device=self.device)
        demand = torch.tensor(
            np.random.randint(1, 10, size=(batch_size, problem_size)),
            dtype=torch.float32,
            device=self.device,
        )
        backhauls = torch.randperm(problem_size, device=self.device)[
            : int(problem_size * self.backhaul_ratio)
        ]
        demand[:, backhauls] *= -1
        return depot, nodes, demand / VRP_CAPACITIES[problem_size]

    def load_problems(self, batch_size, problems=None, aug_factor=1):
        depot, nodes, demand = (
            self.get_random_problems(batch_size, self.problem_size)
            if problems is None
            else problems
        )
        if aug_factor not in (1, 8):
            raise ValueError("Routing augmentation must be 1 or 8")
        if aug_factor == 8:
            depot = augment_xy_data_by_8_fold(depot)
            nodes = augment_xy_data_by_8_fold(nodes)
            demand = demand.repeat(8, 1)
        self.batch_size = len(depot)
        starts = []
        for row in demand:
            positive = torch.nonzero(row > 0, as_tuple=True)[0] + 1
            starts.append(
                positive
                if len(positive)
                else torch.arange(1, self.problem_size + 1, device=self.device)
            )
        self.pomo_size = min(self.requested_pomo_size, min(len(row) for row in starts))
        self.START_NODE = torch.stack([row[: self.pomo_size] for row in starts])
        self.depot_node_xy = torch.cat((depot, nodes), dim=1)
        self.depot_node_demand = torch.cat(
            (demand.new_zeros(self.batch_size, 1), demand), dim=1
        )
        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[
            :, None
        ].expand(-1, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[
            None, :
        ].expand(self.batch_size, -1)
        self.reset_state = Reset_State(depot, nodes, demand)

    def reset(self):
        shape = (self.batch_size, self.pomo_size)
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty(
            *shape, 0, dtype=torch.long, device=self.device
        )
        self.visited = torch.zeros(
            *shape, self.problem_size + 1, dtype=torch.bool, device=self.device
        )
        self.finished = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.route_deliveries = torch.zeros(shape, device=self.device)
        self.route_net_pickups = torch.zeros(shape, device=self.device)
        self.route_peak_net = torch.zeros(shape, device=self.device)
        self.load = torch.ones(shape, device=self.device)
        self.ninf_mask = torch.zeros(*shape, self.problem_size + 1, device=self.device)
        self.step_state = Step_State(self.BATCH_IDX, self.POMO_IDX, self.START_NODE)
        self._update_state()
        return self.reset_state, None, False

    def _update_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.load = self.load

    def pre_step(self):
        return self.step_state, None, False

    def step(self, selected):
        if not torch.isfinite(
            self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected]
        ).all():
            raise ValueError(
                "VRPB action violates a visited-node or route-capacity constraint"
            )
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected.unsqueeze(-1)), dim=-1
        )
        demand = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1)
        chosen = demand.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        self.route_deliveries += chosen.clamp_min(0)
        self.route_net_pickups -= chosen
        self.route_peak_net = torch.maximum(self.route_peak_net, self.route_net_pickups)
        at_depot = selected == 0
        for value in (
            self.route_deliveries,
            self.route_net_pickups,
            self.route_peak_net,
        ):
            value[at_depot] = 0
        self.visited[self.BATCH_IDX, self.POMO_IDX, selected] = True
        self.finished |= self.visited[..., 1:].all(-1) & at_depot
        # If D is total delivery demand in the partial route and M the maximum
        # prefix (pickups-deliveries), ending the route is feasible iff D+M<=1.
        # A candidate delivery increases D; a pickup may increase M. This checks
        # the eventual initial payload as well as every intermediate payload.
        next_deliveries = self.route_deliveries[..., None] + demand.clamp_min(0)
        next_net = self.route_net_pickups[..., None] - demand
        next_peak = torch.maximum(self.route_peak_net[..., None], next_net)
        invalid = self.visited | (next_deliveries + next_peak > 1 + 1e-5)
        invalid[..., 0] = at_depot & ~self.finished
        invalid[self.finished] = True
        invalid[..., 0][self.finished] = False
        self.ninf_mask = demand.new_zeros(demand.shape).masked_fill(
            invalid, float("-inf")
        )
        self.load = 1 - self.route_deliveries - self.route_peak_net
        self._update_state()
        done = self.finished.all()
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _get_travel_distance(self):
        index = self.selected_node_list[..., None].expand(-1, -1, -1, 2)
        ordered = (
            self.depot_node_xy[:, None, :, :]
            .expand(-1, self.pomo_size, -1, -1)
            .gather(2, index)
        )
        return (ordered - ordered.roll(-1, 2)).norm(dim=-1).sum(-1)

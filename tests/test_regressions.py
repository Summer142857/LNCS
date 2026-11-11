"""Regression checks for release constraints, compatibility and input safety."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import torch

from lncs.cli import configuration as cli_configuration, parser, validate_configuration
from lncs.data import validate_dataset
from lncs.engine import (
    generate_dataset,
    load_batch,
    load_weights,
    make_env,
    make_model,
    rollout,
    setup,
    train,
)
from tests.test_release import configuration, validate
from utils.utils import tag_prompt


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.device = setup(17, "cpu")

    def vrpb(self, demands):
        env = make_env("VRPB", 20, 2, self.device)
        demand = torch.zeros(1, 20)
        demand[0, : len(demands)] = torch.tensor(demands)
        env.load_problems(
            1, problems=(torch.zeros(1, 1, 2), torch.rand(1, 20, 2), demand)
        )
        env.reset()
        env.step(torch.zeros(1, env.pomo_size, dtype=torch.long))
        return env

    def action(self, env, node):
        return torch.full((1, env.pomo_size), node, dtype=torch.long)

    def test_vrpb_rejects_excess_initial_deliveries_and_allows_legal_pickup(self):
        env = self.vrpb([0.7, -0.7, 0.7])
        env.step(self.action(env, 1))
        env.step(self.action(env, 2))
        self.assertTrue(torch.isneginf(env.ninf_mask[..., 3]).all())
        with self.assertRaises(ValueError):
            env.step(self.action(env, 3))
        env = self.vrpb([0.1, -0.9])
        env.step(self.action(env, 1))
        self.assertTrue(torch.isfinite(env.ninf_mask[..., 2]).all())
        env.step(self.action(env, 2))
        env.step(self.action(env, 0))
        torch.testing.assert_close(env.load, torch.ones_like(env.load))

    def test_vrpb_masks_match_independent_payload_enumeration(self):
        # Include every candidate pickup/delivery sign pattern, rather than
        # deriving the expected answer from the environment's mask formula.
        for demands in itertools.product((-0.7, -0.3, 0.3, 0.7), repeat=3):
            env = self.vrpb(demands)
            route = []
            for index, demand in enumerate(demands, 1):
                candidate = route + [demand]
                payload = sum(max(value, 0) for value in candidate)
                trajectory = [payload]
                for value in candidate:
                    payload -= value
                    trajectory.append(payload)
                legal = min(trajectory) >= -1e-5 and max(trajectory) <= 1 + 1e-5
                self.assertEqual(
                    bool(torch.isfinite(env.ninf_mask[..., index]).all()),
                    legal,
                    (demands, index),
                )
                if not legal:
                    break
                env.step(self.action(env, index))
                route = candidate

    def test_graphs_support_variable_edges_empty_graphs_and_padding(self):
        model = make_model(configuration(), self.device).eval()
        graphs = [nx.empty_graph(20), nx.path_graph(20), nx.complete_graph(20)]
        data = dict(
            format_version=1,
            task="MVC",
            problem_size=20,
            episodes=3,
            data={
                "adjacency": torch.tensor(
                    [nx.to_numpy_array(g).tolist() for g in graphs], dtype=torch.int8
                )
            },
        )
        with torch.no_grad():
            for task in ("MVC", "MIS"):
                data["task"] = task
                env = make_env(task, 20, 4, self.device)
                load_batch(env, task, 3, data)
                reward, _ = rollout(model, env, task)
                validate(self, env, task, reward)
                self.assertTrue(env.finished.all())
                self.assertEqual(int(env.node_count[0, 0]), 0 if task == "MVC" else 20)

    def test_empty_vertex_cover_training_has_zero_finite_loss(self):
        model = make_model(configuration(), self.device).train()
        env = make_env("MVC", 20, 4, self.device)
        env.FLAG__use_saved_problems = True
        env.saved_problems = [nx.empty_graph(20)]
        env.saved_index = 0
        env.load_problems(1)
        reward, loss = rollout(model, env, "MVC", training=True)
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue((reward == 0).all())
        loss.backward()

    def test_task_cache_uses_exact_metadata_and_is_stable_when_warm(self):
        model = make_model(configuration(), self.device).eval()
        env = make_env("MVC", 20, 4, self.device)
        env.FLAG__use_saved_problems = True
        env.saved_problems = [nx.path_graph(20), nx.complete_graph(20)]
        env.saved_index = 0
        env.load_problems(2)
        state, _, _ = env.reset()
        with torch.no_grad():
            model.pre_forward(state, "MVC")
            first = model.task_embed.clone()
            self.assertEqual(len(model.task_description), 2)
            self.assertFalse(torch.equal(first[0], first[1]))
            model.pre_forward(state, "MVC")
            torch.testing.assert_close(first, model.task_embed, rtol=0, atol=0)

    def test_dataset_validation_accepts_generators_and_rejects_corruption(self):
        for task in ("TSP", "CVRP", "KP", "MVC", "SMTWTP", "VRPB", "MIS"):
            data = generate_dataset(task, 20, 2, 17)
            validate_dataset(data, task, 20, 2)
            with self.assertRaises(ValueError):
                validate_dataset(data, task, 20, 3)
        data = generate_dataset("TSP", 20, 2, 17)
        for replacement in (
            torch.zeros(2, 19, 2),
            torch.full((2, 20, 2), float("nan")),
            torch.full((2, 20, 2), 1.1),
        ):
            data["data"]["problems"] = replacement
            with self.assertRaises(ValueError):
                validate_dataset(data, "TSP", 20, 2)
        graph = generate_dataset("MVC", 20, 2, 17)
        graph["data"]["adjacency"][0, 0, 1] = 1 - graph["data"]["adjacency"][0, 0, 1]
        with self.assertRaises(ValueError):
            validate_dataset(graph, "MVC", 20, 2)

    def test_only_known_unused_legacy_weights_are_ignored(self):
        model = make_model(configuration(), self.device)
        self.assertFalse(
            any(name.startswith("llm_emb") for name, _ in model.named_parameters())
        )
        state = dict(model.state_dict())
        state.update(
            {"llm_emb.weight": torch.zeros(64, 64), "llm_emb.bias": torch.zeros(64)}
        )
        load_weights(model, {"model_state_dict": state})
        state["unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            load_weights(model, {"model_state_dict": state})

    def test_invalid_configs_and_resume_do_not_overwrite_runs(self):
        cfg = configuration()
        for change in (
            {"tasks": []},
            {"tasks": ["TSP", "TSP"]},
            {"learning_rate": float("nan")},
            {"batch_size": 1.5},
        ):
            with self.assertRaises(ValueError):
                validate_configuration(dict(cfg, **change))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            marker = run / "config.json"
            marker.write_text(json.dumps(cfg))
            original = marker.read_bytes()
            with self.assertRaises(FileExistsError):
                train(cfg, run)
            ck = root / "old.pt"
            torch.save({"format_version": 1, "config": cfg}, ck)
            with self.assertRaisesRegex(ValueError, "RC3"):
                train(cfg, run, ck, resume=True)
            self.assertEqual(marker.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            cli_configuration(parser("train").parse_args(["--resume"]))

    def test_cli_inherits_checkpoint_architecture_and_resume_controls(self):
        cfg = configuration()
        cfg.update(tasks=["MIS"], problem_size=50, pomo_size=7, batch_size=3, epochs=11)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ck.pt"
            torch.save({"config": cfg}, path)
            args = parser("evaluate").parse_args(
                ["--checkpoint", str(path), "--data-dir", temp, "--device", "cpu"]
            )
            result = cli_configuration(args, evaluation=True)
            self.assertEqual(result["model"], cfg["model"])
            self.assertEqual(result["tasks"], ["MIS"])
            self.assertEqual(result["problem_size"], 50)
            args = parser("train").parse_args(
                ["--checkpoint", str(path), "--resume", "--epochs", "12"]
            )
            result = cli_configuration(args)
            self.assertEqual(result["batch_size"], 3)
            self.assertEqual(result["pomo_size"], 7)
            self.assertEqual(result["epochs"], 12)

    def test_zero_weight_knapsack_prompt_has_no_nan(self):
        values = torch.zeros(20, 2)
        values[1, 1] = 1
        text = " ".join(tag_prompt(values, "KP"))
        self.assertNotIn("nan", text.lower())


if __name__ == "__main__":
    unittest.main()

"""Offline tests of feasibility, objectives, task caching and CGERL updates."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lncs.engine import (
    ENVS,
    evaluate,
    generate_dataset,
    load_batch,
    make_env,
    make_model,
    rollout,
    setup,
    train,
)
from pcgrad import PCGrad
from utils.utils import tag_prompt


def configuration():
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/base.json").read_text()
    )
    cfg.update(
        encoder="mock",
        device="cpu",
        problem_size=20,
        pomo_size=4,
        episodes=5,
        epochs=1,
        batch_size=2,
        save_every=1,
    )
    cfg["model"].update(
        embedding_dim=64, encoder_layer_num=2, head_num=4, qkv_dim=8, ff_hidden_dim=64
    )
    return cfg


def validate(test, env, task, reward):
    seq = env.selected_node_list.cpu()
    r = reward.cpu()
    n = env.problem_size
    for b in range(env.batch_size):
        for p in range(env.pomo_size):
            route = seq[b, p].tolist()
            if task in ("TSP", "SMTWTP"):
                test.assertEqual(sorted(route), list(range(n)))
                if task == "TSP":
                    xy = env.problems[b, route].cpu()
                    cost = (xy - xy.roll(-1, 0)).norm(dim=1).sum().item()
                else:
                    proc = env.processing_time[b, route].cpu()
                    due = env.due_time[b, route].cpu()
                    weights = env.weights[b, route].cpu()
                    cost = ((proc.cumsum(0) - due).clamp_min(0) * weights).sum().item()
                test.assertAlmostEqual(cost, -r[b, p].item(), places=4)
            elif task == "KP":
                chosen = [x for x in route if x < n]
                test.assertEqual(len(chosen), len(set(chosen)))
                items = env.problems[b, chosen].cpu()
                test.assertLessEqual(
                    items[:, 0].sum().item(), {20: 5.0, 50: 12.5, 100: 25.0}[n] + 1e-5
                )
                test.assertAlmostEqual(
                    items[:, 1].sum().item(), r[b, p].item(), places=4
                )
            elif task in ("MVC", "MIS"):
                count = int(env.node_count[b, p])
                chosen = set(route[:count])
                test.assertEqual(len(chosen), count)
                graph = env.problems[b]
                if task == "MVC":
                    test.assertTrue(
                        all(u in chosen or v in chosen for u, v in graph.edges())
                    )
                    test.assertEqual(-r[b, p].item(), count)
                else:
                    test.assertTrue(
                        all(not (u in chosen and v in chosen) for u, v in graph.edges())
                    )
                    test.assertEqual(r[b, p].item(), count)
            elif task in ("CVRP", "VRPB"):
                customers = [x for x in route if x != 0]
                test.assertEqual(sorted(customers), list(range(1, n + 1)))
                demand = env.depot_node_demand[b].cpu()
                test.assertEqual(route[0], 0)
                test.assertEqual(route[-1], 0)
                trip = []
                for node in route[1:]:
                    if node:
                        trip.append(node)
                        continue
                    load = sum(max(demand[j].item(), 0) for j in trip)
                    test.assertLessEqual(load, 1 + 1e-5)
                    for customer in trip:
                        load -= demand[customer].item()
                        test.assertGreaterEqual(load, -1e-5)
                        test.assertLessEqual(load, 1 + 1e-5)
                    trip = []
                xy = env.depot_node_xy[b, route].cpu()
                cost = (xy - xy.roll(-1, 0)).norm(dim=1).sum().item()
                test.assertAlmostEqual(cost, -r[b, p].item(), places=4)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.device = setup(1234, "cpu")

    def test_pcgrad_projects_and_preserves_reference_order(self):
        param = torch.nn.Parameter(torch.ones(2))
        pc = PCGrad(torch.optim.SGD([param], lr=0.1))
        grads = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])]
        copies = [g.clone() for g in grads]
        mask = [torch.ones(2, dtype=torch.bool)] * 2
        result = pc._project_conflicting(grads, mask)
        torch.testing.assert_close(result, torch.tensor([0.5, 1.5]))
        for before, after in zip(copies, grads):
            torch.testing.assert_close(before, after)
        result = pc._project_conflicting([torch.zeros(2), grads[0]], mask)
        test = torch.isfinite(result).all().item()
        self.assertTrue(test)

    def test_partial_gradient_parameters_and_adam_update(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        unused = torch.nn.Parameter(torch.ones(2))
        pc = PCGrad(torch.optim.Adam([p, unused], lr=1e-3))
        losses = [p[0], -p[0] + p[1]]
        pc.pc_backward(losses)
        torch.testing.assert_close(p.grad, torch.tensor([0.5, 1.5]))
        self.assertIsNone(unused.grad)
        pc.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_prompts_include_last_job_and_transfer_instructions(self):
        due = torch.arange(20).float() + 0.1
        weights = torch.arange(20).float() + 0.2
        proc = torch.arange(20).float() + 0.3
        prompts = tag_prompt((due, weights, proc), "SMTWTP")
        self.assertEqual(len(prompts), 21)
        self.assertIn("Node (19)", prompts[-1])
        self.assertIn("19.3", prompts[-1])
        for task in ("MIS", "VRPB"):
            env = make_env(task, 20, 4, self.device)
            env.load_problems(1)
            state, _, _ = env.reset()
            instance = (
                state.problems[0]
                if task == "MIS"
                else (state.depot_xy[0], state.node_xy[0], state.node_demand[0])
            )
            self.assertEqual(
                len(tag_prompt(instance, task)), 21 if task == "MIS" else 22
            )

    def test_all_seven_task_solutions_are_feasible_and_rewards_correct(self):
        model = make_model(configuration(), self.device).eval()
        with torch.no_grad():
            for task in ENVS:
                with self.subTest(task=task):
                    env = make_env(task, 20, 4, self.device)
                    load_batch(env, task, 2)
                    reward, _ = rollout(model, env, task)
                    validate(self, env, task, reward)

    def test_cached_task_embedding_handles_changing_batch_and_augmentation(self):
        model = make_model(configuration(), self.device).eval()
        with torch.no_grad():
            for task in ENVS:
                for batch in (1, 3, 2):
                    env = make_env(task, 20, 4, self.device)
                    aug = 8 if task in ("TSP", "CVRP", "VRPB") else 1
                    load_batch(env, task, batch, augmentation=aug)
                    reward, _ = rollout(model, env, task)
                    self.assertEqual(reward.shape[0], batch * aug)
                    self.assertTrue(torch.isfinite(reward).all())
                    self.assertTrue(
                        all(
                            value.shape[0] == 1
                            for value in model.task_description.values()
                        )
                    )

    def test_supported_sizes_50_and_100(self):
        model = make_model(configuration(), self.device).eval()
        with torch.no_grad():
            for size in (50, 100):
                for task in ENVS:
                    with self.subTest(size=size, task=task):
                        env = make_env(task, size, 4, self.device)
                        load_batch(env, task, 1)
                        reward, _ = rollout(model, env, task)
                        validate(self, env, task, reward)

    def test_dataset_reproducibility(self):
        for task in ENVS:
            a = generate_dataset(task, 20, 3, 41)
            b = generate_dataset(task, 20, 3, 41)
            for key in a["data"]:
                torch.testing.assert_close(a["data"][key], b["data"][key])

    def test_train_resume_matches_uninterrupted_and_evaluation(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train(cfg, root / "split")
            cfg["epochs"] = 2
            train(cfg, root / "split", root / "split/checkpoint-1.pt", resume=True)
            train(cfg, root / "full")
            a = torch.load(root / "split/checkpoint-2.pt", weights_only=True)
            b = torch.load(root / "full/checkpoint-2.pt", weights_only=True)
            for key in a["model_state_dict"]:
                torch.testing.assert_close(
                    a["model_state_dict"][key],
                    b["model_state_dict"][key],
                    rtol=0,
                    atol=0,
                )
            (root / "data").mkdir()
            cfg["tasks"] = list(ENVS)
            for task in cfg["tasks"]:
                torch.save(
                    generate_dataset(task, 20, 5, 123),
                    root / f"data/{task.lower()}_20.pt",
                )
            report = evaluate(
                cfg, root / "full/checkpoint-2.pt", root / "data", root / "report.json"
            )
            for result in report["results"].values():
                self.assertEqual(result["instances"], 5)
                self.assertTrue(np.isfinite(result["mean"]))

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable")
    def test_mps_resume_preserves_stochastic_state(self):
        cfg = configuration()
        cfg.update(device="mps", tasks=["TSP"], episodes=3)
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                train(cfg, root / "split")
                saved = torch.load(root / "split/checkpoint-1.pt", weights_only=True)
                self.assertIsNotNone(saved["rng_mps"])
                cfg["epochs"] = 2
                train(cfg, root / "split", root / "split/checkpoint-1.pt", resume=True)
                train(cfg, root / "full")
                a = torch.load(root / "split/checkpoint-2.pt", weights_only=True)
                b = torch.load(root / "full/checkpoint-2.pt", weights_only=True)
                torch.testing.assert_close(a["rng_mps"], b["rng_mps"], rtol=0, atol=0)
                for key in a["model_state_dict"]:
                    torch.testing.assert_close(
                        a["model_state_dict"][key],
                        b["model_state_dict"][key],
                        rtol=1e-4,
                        atol=1e-6,
                    )
        finally:
            setup(1234, "cpu")


if __name__ == "__main__":
    unittest.main()

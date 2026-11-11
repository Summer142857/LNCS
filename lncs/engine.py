"""Shared training/evaluation and portable synthetic dataset support."""

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from UnifiedModel import UnifiedModel
from TSP.POMO.TSPEnv import TSPEnv
from CVRP.POMO.CVRPEnv import CVRPEnv
from KP.POMO.KPEnv import KPEnv
from MVC.POMO.MVCEnv import MVCEnv
from SMTWTP.POMO.SMTWTPEnv import SMTWTPEnv
from MIS.MISEnv import MISEnv
from VRPB.VRPBEnv import VRPBEnv
from pcgrad import PCGrad
from lncs.data import validate_dataset

ENVS = dict(
    TSP=TSPEnv,
    CVRP=CVRPEnv,
    KP=KPEnv,
    MVC=MVCEnv,
    SMTWTP=SMTWTPEnv,
    MIS=MISEnv,
    VRPB=VRPBEnv,
)
MAXIMIZE = {"KP", "MIS"}
ROUTING = {"TSP", "CVRP", "VRPB"}


def setup(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; use --device cpu for software tests."
            )
        torch.cuda.set_device(target)
        torch.cuda.manual_seed_all(seed)
    elif target.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS requested but unavailable; use --device cpu.")
        torch.mps.manual_seed(seed)
    # Research environments create tensors without explicit device arguments.
    torch.set_default_device(target)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(2)
    return target


def make_env(task, size, pomo_size, device):
    return ENVS[task](
        problem_size=size,
        pomo_size=pomo_size,
        FLAG__use_saved_problems=False,
        device=device,
    )


def make_model(config, device):
    params = dict(config["model"])
    params.update(
        llm=config["encoder"],
        text_encoder=True,
        eval_type="argmax",
        sqrt_embedding_dim=params["embedding_dim"] ** 0.5,
        device=str(device),
        cache_dir=config.get("cache_dir"),
        local_files_only=config.get("local_files_only", False),
        pooling=config.get("pooling", "legacy_mean"),
        llm_dtype=config.get("llm_dtype", "float32"),
    )
    return UnifiedModel(**params).to(device)


def generate_dataset(task, size, episodes, seed):
    setup(seed, "cpu")
    env = make_env(task, size, size, torch.device("cpu"))
    env.load_problems(episodes)
    if task in ("MVC", "MIS"):
        data = {
            "adjacency": torch.stack(
                [
                    torch.from_numpy(nx.to_numpy_array(g, dtype=np.int8))
                    for g in env.problems
                ]
            )
        }
    elif task in ("CVRP", "VRPB"):
        state = env.reset_state
        data = dict(
            depot_xy=state.depot_xy,
            node_xy=state.node_xy,
            node_demand=state.node_demand,
        )
    elif task == "SMTWTP":
        data = dict(
            due_time=env.due_time,
            weights=env.weights,
            processing_time=env.processing_time,
        )
    else:
        data = {"problems": env.problems}
    return dict(
        format_version=1,
        task=task,
        problem_size=size,
        episodes=episodes,
        seed=seed,
        data=data,
    )


def read_dataset(path, task, size, episodes):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return validate_dataset(data, task, size, episodes)


def load_batch(env, task, batch_size, dataset=None, offset=0, augmentation=1):
    if dataset is None:
        if task in ROUTING:
            env.load_problems(batch_size, aug_factor=augmentation)
        else:
            env.load_problems(batch_size)
        return
    device = env.env_params["device"]
    values = {
        key: val[offset : offset + batch_size].to(device)
        for key, val in dataset["data"].items()
    }
    if task == "VRPB":
        env.load_problems(
            batch_size,
            problems=tuple(
                values[key] for key in ("depot_xy", "node_xy", "node_demand")
            ),
            aug_factor=augmentation,
        )
        return
    env.FLAG__use_saved_problems = True
    env.saved_index = 0
    if task in ("MVC", "MIS"):
        env.saved_problems = [
            nx.from_numpy_array(adj.cpu().numpy()) for adj in values["adjacency"]
        ]
    elif task == "CVRP":
        for key, val in values.items():
            setattr(env, "saved_" + key, val)
    elif task == "SMTWTP":
        for key, val in values.items():
            setattr(env, "saved_" + key, val)
    else:
        env.saved_problems = values["problems"]
    if task in ROUTING:
        env.load_problems(batch_size, aug_factor=augmentation)
    else:
        env.load_problems(batch_size)


def rollout(model, env, task, training=False):
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state, problem=task)
    state, reward, done = env.pre_step()
    logs = []
    steps = 0
    while not bool(done):
        finished = getattr(state, "finished", None)
        finished = finished.clone() if finished is not None else None
        selected, probability = model(state)
        action = selected.clone()
        if task == "KP" and finished is not None:
            action[finished] = env.problem_size  # zero-weight, zero-profit dummy
        state, reward, done = env.step(action)
        if training:
            if finished is not None:
                probability = torch.where(
                    finished, torch.ones_like(probability), probability
                )
            logs.append(
                probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
            )
        steps += 1
        if steps > 4 * env.problem_size + 10:
            raise RuntimeError(f"{task} did not terminate; check the action masks.")
    if training:
        advantage = reward - reward.mean(dim=1, keepdim=True)
        loss = (
            -(advantage * torch.stack(logs).sum(0)).mean()
            if logs
            else model.encoded_nodes.sum() * 0
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {task} training loss")
        return reward, loss
    return reward, None


def objective_values(reward, task, batch_size, augmentation=1):
    best = reward.reshape(augmentation, batch_size, -1).max(2).values.max(0).values
    return best if task in MAXIMIZE else -best


def load_weights(model, path):
    checkpoint = (
        path
        if isinstance(path, dict)
        else torch.load(path, map_location="cpu", weights_only=True)
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("_orig_mod."): val for key, val in state.items()}
    # RC1/RC2 included an unused connector copy. No other unexpected weights
    # are ignored; all active generator parameters still load strictly.
    state = {
        key: val
        for key, val in state.items()
        if key not in ("llm_emb.weight", "llm_emb.bias")
    }
    model.load_state_dict(state, strict=True)
    # Recompute task embeddings. Historical files cache an entire training batch;
    # their row count and task set can differ from the current evaluation.
    model.task_description.clear()
    return checkpoint


def save_checkpoint(path, model, optimizer, config, epoch):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        dict(
            format_version=2,
            epoch=epoch,
            model_state_dict=state,
            optimizer_state_dict=optimizer.state_dict(),
            config=config,
            rng_torch=torch.get_rng_state(),
            rng_python=random.getstate(),
            rng_numpy=list(np.random.get_state()[:1])
            + [torch.from_numpy(np.random.get_state()[1].astype(np.int64))]
            + list(np.random.get_state()[2:]),
            task_embed={
                key: val.detach().cpu() for key, val in model.task_description.items()
            },
            rng_mps=torch.mps.get_rng_state()
            if torch.device(config["device"]).type == "mps"
            else None,
            rng_cuda=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        ),
        path,
    )


def train(config, output, checkpoint_path=None, resume=False, overwrite=False):
    from lncs.cli import validate_configuration

    validate_configuration(config)
    if resume and checkpoint_path is None:
        raise ValueError("--resume requires --checkpoint")
    output = Path(output)
    owned_files = ("config.json", "metrics.jsonl")
    if (
        not resume
        and not overwrite
        and (
            any((output / name).exists() for name in owned_files)
            or any(output.glob("checkpoint-*.pt"))
        )
    ):
        raise FileExistsError(
            "Output already contains a training run; use another directory, --resume, or --overwrite"
        )
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint_path
        else None
    )
    start = 1
    if resume:
        if checkpoint.get("format_version") != 2:
            raise ValueError(
                "Resume requires an RC3 checkpoint; use --checkpoint alone to warm-start earlier weights"
            )
        prior = checkpoint["config"]
        for key in (
            "encoder",
            "model",
            "pooling",
            "llm_dtype",
            "problem_size",
            "tasks",
            "pcgrad",
            "episodes",
            "batch_size",
            "pomo_size",
            "learning_rate",
            "weight_decay",
        ):
            if prior[key] != config[key]:
                raise ValueError(
                    f"Cannot resume with changed {key}; omit --resume to fine-tune"
                )
        if torch.device(prior["device"]).type != torch.device(config["device"]).type:
            raise ValueError("Use a warm-start when changing device backends")
        if (output / "config.json").exists():
            existing = json.loads((output / "config.json").read_text())
            for key in (
                "encoder",
                "model",
                "pooling",
                "problem_size",
                "tasks",
                "pcgrad",
                "episodes",
                "batch_size",
                "pomo_size",
            ):
                if existing[key] != prior[key]:
                    raise ValueError(
                        "Output belongs to a different training configuration"
                    )
        start = checkpoint["epoch"] + 1
        if any(
            int(path.stem.split("-")[-1]) > checkpoint["epoch"]
            for path in output.glob("checkpoint-*.pt")
            if path.stem.split("-")[-1].isdigit()
        ):
            raise ValueError(
                "Output contains later checkpoints; resume into a new directory"
            )
        if (output / "metrics.jsonl").exists():
            records = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if any(record["epoch"] > checkpoint["epoch"] for record in records):
                raise ValueError(
                    "Output contains later metrics; resume into a new directory"
                )
    if start > config["epochs"]:
        raise ValueError(
            "No training remains: --epochs is the final epoch number when resuming"
        )
    device = setup(config["seed"], config["device"])
    model = make_model(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    if checkpoint is not None:
        load_weights(model, checkpoint)
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.task_description = {
            key: val.to(device) for key, val in checkpoint["task_embed"].items()
        }
        torch.set_rng_state(checkpoint["rng_torch"])
        random.setstate(checkpoint["rng_python"])
        rng = checkpoint["rng_numpy"]
        np.random.set_state((rng[0], rng[1].numpy().astype(np.uint32), *rng[2:]))
        if checkpoint["rng_cuda"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        if device.type == "mps":
            if checkpoint.get("rng_mps") is None:
                raise ValueError(
                    "Checkpoint has no MPS random state; omit --resume to warm-start it."
                )
            torch.mps.set_rng_state(checkpoint["rng_mps"])
    output.mkdir(parents=True, exist_ok=True)
    if overwrite and not resume:
        for path in output.glob("checkpoint-*.pt"):
            path.unlink()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    update = PCGrad(optimizer, reduction="sum") if config["pcgrad"] else optimizer
    envs = {
        task: make_env(task, config["problem_size"], config["pomo_size"], device)
        for task in config["tasks"]
    }
    model.train()
    history_path = output / "metrics.jsonl"
    with history_path.open("a" if resume else "w") as history:
        for epoch in range(start, config["epochs"] + 1):
            scores = {task: [] for task in envs}
            losses_record = []
            for offset in range(0, config["episodes"], config["batch_size"]):
                batch = min(config["batch_size"], config["episodes"] - offset)
                losses = []
                for task, env in envs.items():
                    load_batch(env, task, batch)
                    reward, loss = rollout(model, env, task, training=True)
                    scores[task].extend(
                        objective_values(reward, task, batch).detach().cpu().tolist()
                    )
                    losses.append(loss)
                update.zero_grad()
                if config["pcgrad"]:
                    update.pc_backward(losses)
                else:
                    torch.stack(losses).sum().backward()
                update.step()
                losses_record.append(sum(loss.detach().item() for loss in losses))
            record = dict(
                epoch=epoch,
                loss=float(np.mean(losses_record)),
                objective={
                    task: float(np.mean(values)) for task, values in scores.items()
                },
            )
            print(json.dumps(record), flush=True)
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch % config["save_every"] == 0 or epoch == config["epochs"]:
                save_checkpoint(
                    output / f"checkpoint-{epoch}.pt", model, optimizer, config, epoch
                )
    return output


@torch.no_grad()
def evaluate(config, checkpoint_path, data_dir, output, augmentation=8):
    from lncs.cli import validate_configuration

    validate_configuration(config)
    if augmentation not in (1, 8):
        raise ValueError("Augmentation must be 1 or 8")
    device = setup(config["seed"], config["device"])
    datasets = {
        task: read_dataset(
            Path(data_dir) / f"{task.lower()}_{config['problem_size']}.pt",
            task,
            config["problem_size"],
            config["episodes"],
        )
        for task in config["tasks"]
    }
    model = make_model(config, device)
    load_weights(model, checkpoint_path)
    model.eval()
    results = {}
    for task, dataset in datasets.items():
        env = make_env(task, config["problem_size"], config["pomo_size"], device)
        values = []
        aug = augmentation if task in ROUTING else 1
        for offset in range(0, config["episodes"], config["batch_size"]):
            batch = min(config["batch_size"], config["episodes"] - offset)
            load_batch(env, task, batch, dataset, offset, aug)
            reward, _ = rollout(model, env, task)
            values.extend(objective_values(reward, task, batch, aug).cpu().tolist())
        results[task] = dict(
            mean=float(np.mean(values)),
            standard_error=float(np.std(values, ddof=1) / len(values) ** 0.5)
            if len(values) > 1
            else 0.0,
            instances=len(values),
            objective_direction="maximize" if task in MAXIMIZE else "minimize",
            augmentation=aug,
        )
    report = dict(
        config=config,
        checkpoint=str(checkpoint_path),
        results=results,
        note="Raw objective values. No optimality gap is claimed without certified reference solutions.",
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return report

"""Command-line configuration, kept separate from the numerical engine."""

import argparse
import json
import math
from pathlib import Path

from lncs.problems import TASKS, SIZES

ENCODERS = ("llama2_7b", "llama3_8b", "qwen3", "ST", "e5", "mock")


def parser(mode):
    p = argparse.ArgumentParser(description=f"LNCS {mode}")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/base.json",
    )
    p.add_argument("--tasks", nargs="+", choices=TASKS)
    p.add_argument("--problem-size", type=int, choices=SIZES)
    p.add_argument("--pomo-size", type=int)
    p.add_argument("--encoder", choices=ENCODERS)
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--cache-dir")
    p.add_argument("--local-files-only", action="store_true", default=None)
    p.add_argument("--pooling", choices=("legacy_mean", "masked_mean"))
    p.add_argument("--llm-dtype", choices=("float32", "float16", "bfloat16"))
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Small mock-encoder software test; no paper performance claims",
    )
    p.add_argument("--checkpoint", type=Path, required=mode == "evaluate")
    if mode == "train":
        p.add_argument("--epochs", type=int)
        p.add_argument("--learning-rate", type=float)
        p.add_argument("--no-pcgrad", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument(
            "--overwrite",
            action="store_true",
            help="Allow reuse of a training output directory",
        )
        p.add_argument("--output", type=Path, default=Path("outputs/train"))
    else:
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--augmentation", type=int, choices=(1, 8), default=8)
        p.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    return p


def configuration(args, evaluation=False):
    cfg = json.loads(args.config.read_text())
    if getattr(args, "resume", False) and args.checkpoint is None:
        raise ValueError("--resume requires --checkpoint")
    if args.checkpoint is not None:
        import torch

        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "config" in ck:
            keys = ["encoder", "model", "pooling", "llm_dtype"]
            if evaluation or getattr(args, "resume", False):
                keys += ["problem_size", "pomo_size", "tasks", "device"]
            if getattr(args, "resume", False):
                keys += [
                    "seed",
                    "device",
                    "epochs",
                    "episodes",
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                    "pcgrad",
                    "save_every",
                ]
            for key in keys:
                if key in ck["config"]:
                    cfg[key] = ck["config"][key]
        elif args.encoder is None:
            raise ValueError(
                "A checkpoint without configuration requires an explicit --encoder"
            )
    if args.smoke_test:
        if args.checkpoint is not None and cfg["encoder"] != "mock":
            raise ValueError("--smoke-test cannot load a real-encoder checkpoint")
        cfg.update(
            encoder="mock",
            device="cpu",
            problem_size=20,
            pomo_size=4,
            epochs=1,
            episodes=4,
            batch_size=2,
            save_every=1,
        )
        cfg["model"].update(
            embedding_dim=64,
            encoder_layer_num=2,
            ff_hidden_dim=64,
            head_num=4,
            qkv_dim=8,
        )
    if args.problem_size is not None and args.pomo_size is None:
        cfg["pomo_size"] = args.problem_size
        if not evaluation and args.epochs is None:
            cfg["epochs"] = 100 if args.problem_size == 20 else 200
    for key in (
        "tasks",
        "problem_size",
        "pomo_size",
        "encoder",
        "device",
        "seed",
        "episodes",
        "batch_size",
        "cache_dir",
        "local_files_only",
        "pooling",
        "llm_dtype",
        "epochs",
        "learning_rate",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if evaluation and args.episodes is None:
        cfg["episodes"] = 4 if args.smoke_test else 1000
    if getattr(args, "no_pcgrad", False):
        cfg["pcgrad"] = False
    validate_configuration(cfg)
    return cfg


def validate_configuration(cfg):
    tasks = cfg["tasks"]
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(task not in TASKS for task in tasks)
    ):
        raise ValueError("tasks must be a nonempty list of supported task names")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate tasks are not allowed")
    if type(cfg["problem_size"]) is not int or cfg["problem_size"] not in SIZES:
        raise ValueError(f"problem_size must be one of {SIZES}")
    if cfg["encoder"] not in ENCODERS:
        raise ValueError("Unsupported encoder")
    if cfg.get("pooling", "legacy_mean") not in ("legacy_mean", "masked_mean"):
        raise ValueError("Unsupported pooling mode")
    if cfg.get("llm_dtype", "float32") not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported encoder dtype")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("episodes", "epochs", "batch_size", "pomo_size", "save_every"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 1 < cfg["pomo_size"] <= cfg["problem_size"]:
        raise ValueError("pomo_size must be between 2 and problem_size")
    for key in ("learning_rate", "weight_decay"):
        value = cfg[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if type(cfg["pcgrad"]) is not bool:
        raise ValueError("pcgrad must be boolean")
    if cfg["device"].split(":")[0] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda[:index] or mps")
    keys = {
        "embedding_dim",
        "encoder_layer_num",
        "qkv_dim",
        "head_num",
        "ff_hidden_dim",
        "logit_clipping",
    }
    if set(cfg["model"]) != keys:
        raise ValueError(f"model must contain exactly {sorted(keys)}")
    for key, value in cfg["model"].items():
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"model.{key} must be positive and finite")
        if key != "logit_clipping" and type(value) is not int:
            raise ValueError(f"model.{key} must be an integer")


def train_main():
    args = parser("train").parse_args()
    cfg = configuration(args)
    from lncs.engine import train

    train(cfg, args.output, args.checkpoint, args.resume, args.overwrite)


def evaluate_main():
    args = parser("evaluate").parse_args()
    cfg = configuration(args, evaluation=True)
    from lncs.engine import evaluate

    evaluate(cfg, args.checkpoint, args.data_dir, args.output, args.augmentation)

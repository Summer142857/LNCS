"""Generate seeded synthetic data with no solver, credential or LLM dependency."""

import argparse
from pathlib import Path
from lncs.problems import TASKS, SIZES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=["TSP", "CVRP", "KP", "MVC", "SMTWTP"],
    )
    p.add_argument("--problem-size", type=int, choices=SIZES, default=50)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        p.error("Duplicate tasks are not allowed")
    if not 0 <= args.seed < 2**32 - len(args.tasks):
        p.error("--seed is out of range")
    paths = [
        args.output / f"{task.lower()}_{args.problem_size}.pt" for task in args.tasks
    ]
    if not args.overwrite and any(path.exists() for path in paths):
        p.error(
            "A dataset already exists; choose another directory or pass --overwrite."
        )
    from lncs.engine import generate_dataset
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    for index, (task, path) in enumerate(zip(args.tasks, paths)):
        torch.save(
            generate_dataset(task, args.problem_size, args.episodes, args.seed + index),
            path,
        )
        print(path)


if __name__ == "__main__":
    main()

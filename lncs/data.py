"""Validate the tensor-only synthetic dataset format before decoding."""

import torch
from lncs.problems import SIZES, TASKS


def validate_dataset(dataset, task, size, episodes):
    if task not in TASKS or size not in SIZES:
        raise ValueError("Unsupported dataset task or problem size")
    if (
        not isinstance(dataset, dict)
        or dataset.get("format_version") != 1
        or dataset.get("task") != task
        or dataset.get("problem_size") != size
    ):
        raise ValueError(
            "Dataset task, size or format does not match the requested evaluation"
        )
    if task in ("MVC", "MIS"):
        shapes = {"adjacency": (size, size)}
    elif task in ("CVRP", "VRPB"):
        shapes = {"depot_xy": (1, 2), "node_xy": (size, 2), "node_demand": (size,)}
    elif task == "SMTWTP":
        shapes = {key: (size,) for key in ("due_time", "weights", "processing_time")}
    else:
        shapes = {"problems": (size, 2)}
    data = dataset.get("data")
    if not isinstance(data, dict) or set(data) != set(shapes):
        raise ValueError(f"Dataset feature keys must be {sorted(shapes)}")
    counts = []
    for key, shape in shapes.items():
        tensor = data[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != len(shape) + 1
            or tuple(tensor.shape[1:]) != shape
        ):
            raise ValueError(f"Dataset {key} must have shape (instances, {shape})")
        if key != "adjacency" and tensor.dtype != torch.float32:
            raise ValueError(f"Dataset {key} must use float32")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Dataset {key} contains non-finite values")
        counts.append(tensor.shape[0])
    if (
        len(set(counts)) != 1
        or not counts
        or counts[0] < episodes
        or dataset.get("episodes") != counts[0]
    ):
        raise ValueError(
            "Dataset instance count is inconsistent or smaller than requested"
        )
    for key, tensor in data.items():
        low, high = (
            (-1, 1)
            if task == "VRPB" and key == "node_demand"
            else (0, size)
            if key == "due_time"
            else (0, 1)
        )
        if ((tensor < low) | (tensor > high)).any():
            raise ValueError(f"Dataset {key} must lie in [{low}, {high}]")
    if task in ("MVC", "MIS"):
        adjacency = data["adjacency"]
        if (
            not ((adjacency == 0) | (adjacency == 1)).all()
            or not torch.equal(adjacency, adjacency.transpose(-1, -2))
            or adjacency.diagonal(dim1=-2, dim2=-1).any()
        ):
            raise ValueError(
                "Dataset adjacency must describe a simple undirected graph"
            )
    return dataset

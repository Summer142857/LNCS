"""CGERL/PCGrad projection; based on Wei-Cheng Tseng's BSD-3-Clause implementation.

See licenses/Pytorch-PCGrad-BSD-3-Clause.txt and RELEASE_NOTES.md.
"""

import random

import torch


class PCGrad:
    def __init__(self, optimizer, reduction="sum"):
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be sum or mean")
        self._optim, self._reduction = optimizer, reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        self._optim.zero_grad(set_to_none=True)

    def step(self):
        self._optim.step()

    def pc_backward(self, objectives):
        if not objectives:
            raise ValueError("At least one task loss is required")
        params = [
            p
            for group in self._optim.param_groups
            for p in group["params"]
            if p.requires_grad
        ]
        gradients, present = [], []
        for loss in objectives:
            self.zero_grad()
            loss.backward()
            gradients.append(
                torch.cat(
                    [
                        (
                            p.grad if p.grad is not None else torch.zeros_like(p)
                        ).flatten()
                        for p in params
                    ]
                )
            )
            present.append(
                torch.cat(
                    [
                        torch.full_like(
                            p, p.grad is not None, dtype=torch.bool
                        ).flatten()
                        for p in params
                    ]
                )
            )
        merged = self._project_conflicting(gradients, present)
        offset = 0
        for p in params:
            count = p.numel()
            # Do not apply Adam weight decay to parameters unused by every task.
            active = torch.stack(present)[:, offset : offset + count].any().item()
            p.grad = (
                merged[offset : offset + count].view_as(p).clone() if active else None
            )
            offset += count

    def _project_conflicting(self, grads, has_grads):
        projected = [g.clone() for g in grads]
        for i, target in enumerate(projected):
            others = [j for j in range(len(grads)) if j != i]
            random.shuffle(others)
            for j in others:
                reference = grads[j]
                dot = torch.dot(target, reference)
                norm_sq = torch.dot(reference, reference)
                if dot < 0 and norm_sq > 0:
                    target.sub_(dot * reference / norm_sq)
        merged = torch.stack(projected).sum(0)
        if self._reduction == "mean":
            shared = torch.stack(has_grads).all(0)
            merged[shared] /= len(grads)
        return merged

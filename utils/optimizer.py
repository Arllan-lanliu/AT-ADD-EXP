"""
Sharpness-Aware Minimisation (SAM) optimizer and BatchNorm helpers.

References
----------
- SAM: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
  Generalization", ICLR 2021.
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm


# ---------------------------------------------------------------------------
# BatchNorm running-stats helpers (used by the SAM two-step update)
# ---------------------------------------------------------------------------

def disable_running_stats(model: torch.nn.Module) -> None:
    """Set momentum=0 on all BatchNorm layers to freeze running stats."""
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model: torch.nn.Module) -> None:
    """Restore momentum on all BatchNorm layers from the saved backup."""
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


# ---------------------------------------------------------------------------
# SAM optimizer
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimisation optimizer.

    Wraps any base optimizer and performs a two-step update:
    1. ``first_step``  — perturb weights to the local loss maximum.
    2. ``second_step`` — restore weights and take the real gradient step.

    Parameters
    ----------
    params:
        Model parameters (same as any ``torch.optim.Optimizer``).
    base_optimizer:
        Underlying optimizer class (e.g. ``torch.optim.Adam``).
    rho:
        Neighbourhood size for the perturbation.
    adaptive:
        If True, use element-wise adaptive scaling (ASAM variant).
    **kwargs:
        Forwarded to ``base_optimizer``.
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 adaptive: bool = False, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, \
            "SAM requires a closure, but none was provided."
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                .norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

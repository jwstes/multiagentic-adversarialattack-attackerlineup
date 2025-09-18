import torch
import torch.nn.functional as F
from .base import MethodAgentBase
from .filters import apply_frequency

class CW_Agent(MethodAgentBase):
    """
    CW-like targeted attack with L2 penalty, L_inf projection, optional frequency shaping.
    Numerically stabilized to avoid NaNs.
    """
    def __init__(self, desk, advisor=None, llm_every_k: int = 5):
        super().__init__("CW_Agent", desk, advisor=advisor, llm_every_k=llm_every_k)

    def generate(self, x: torch.Tensor, target_class: int, epsilon_max: float, strategy: dict | None = None, tv_weight: float = 0.0, roi_mask=None, **kwargs):
        steps = int((strategy or {}).get("steps", 30))
        lr = float((strategy or {}).get("lr", 0.005))               # lower default lr
        l2_weight = float((strategy or {}).get("l2_weight", 0.5))   # moderate L2
        freq = (strategy or {}).get("frequency", "neutral")

        x_orig = x.clone().detach()
        x_adv = x.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x_adv], lr=lr, eps=1e-8)

        for _ in range(max(1, steps)):
            opt.zero_grad(set_to_none=True)

            # Composite loss from base (surrogates CE [+ optional final model CE] [+ TV])
            loss_fn, _ = self._build_loss(x_adv, target_class, tv_weight, roi_mask)
            ce_loss = loss_fn(x_adv)

            delta = x_adv - x_orig
            # use reshape to support non-contiguous memory
            l2 = delta.reshape(delta.size(0), -1).pow(2).sum(dim=1).sqrt().mean()
            loss = ce_loss + l2_weight * l2

            if not torch.isfinite(loss):
                # If something went off the rails, bail out gracefully
                break

            loss.backward()

            # ROI gradient shaping
            if roi_mask is not None and x_adv.grad is not None:
                with torch.no_grad():
                    x_adv.grad.mul_(roi_mask)

            # Gradient clipping to avoid explosions
            torch.nn.utils.clip_grad_norm_([x_adv], max_norm=5.0)

            opt.step()

            # Projection + clamp + NaN guard
            with torch.no_grad():
                # Replace NaNs/Infs before projection
                x_adv.copy_(torch.nan_to_num(x_adv, nan=0.5, posinf=1.0, neginf=0.0))
                # L_inf projection
                delta = (x_adv - x_orig).clamp(min=-epsilon_max, max=epsilon_max)
                # Optional frequency shaping
                delta = apply_frequency(delta, freq)
                # Recompose and clamp to [0,1]
                x_adv.copy_((x_orig + delta).clamp(0.0, 1.0))

        params = {
            "method": "CW",
            "steps": int(steps),
            "lr": float(lr),
            "l2_weight": float(l2_weight),
            "target_class": target_class,
            "frequency": freq,
            "tv_weight": float(tv_weight),
        }
        return x_adv.detach(), params
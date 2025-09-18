import torch
import torch.nn.functional as F
from .base import MethodAgentBase
from .filters import apply_frequency

class PGD_Agent(MethodAgentBase):
    def __init__(self, desk, advisor=None, llm_every_k: int = 5):
        super().__init__("PGD_Agent", desk, advisor=advisor, llm_every_k=llm_every_k)

    def generate(self, x: torch.Tensor, target_class: int, epsilon_max: float, strategy: dict | None = None, tv_weight: float = 0.0, roi_mask=None, **kwargs):
        steps = int((strategy or {}).get("steps", 10))
        eps = float(min(epsilon_max, (strategy or {}).get("epsilon", 12/255)))
        alpha = float((strategy or {}).get("alpha", eps / 4.0))
        freq = (strategy or {}).get("frequency", "neutral")

        x_orig = x.clone().detach()
        x_adv = x.clone().detach()

        for _ in range(max(1, steps)):
            x_adv.requires_grad_(True)
            loss_fn, _ = self._build_loss(x_adv, target_class, tv_weight, roi_mask)
            loss = loss_fn(x_adv)
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad
                if roi_mask is not None:
                    grad = grad * roi_mask
                delta = x_adv - x_orig
                delta = delta - alpha * grad.sign()  # targeted
                delta = apply_frequency(delta, freq)
                delta = delta.clamp(min=-eps, max=eps)
                x_adv = (x_orig + delta).clamp(0.0, 1.0)

        params = {
            "method": "PGD",
            "epsilon": float(eps),
            "alpha": float(alpha),
            "steps": int(steps),
            "target_class": target_class,
            "frequency": freq,
            "tv_weight": float(tv_weight),
        }
        return x_adv.detach(), params
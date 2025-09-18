import torch
import torch.nn.functional as F
from .base import MethodAgentBase
from .filters import apply_frequency

class FGSM_Agent(MethodAgentBase):
    def __init__(self, desk, advisor=None, llm_every_k: int = 5):
        super().__init__("FGSM_Agent", desk, advisor=advisor, llm_every_k=llm_every_k)

    def generate(self, x: torch.Tensor, target_class: int, epsilon_max: float, strategy: dict | None = None, tv_weight: float = 0.0, roi_mask=None, **kwargs):
        eps = min(epsilon_max, float(strategy.get("epsilon", 8/255))) if strategy else min(epsilon_max, 8/255)
        freq = (strategy or {}).get("frequency", "neutral")

        x_orig = x.clone().detach()
        x_adv = x.clone().detach().requires_grad_(True)

        # Composite loss
        loss_fn, _ = self._build_loss(x_adv, target_class, tv_weight, roi_mask)
        loss = loss_fn(x_adv)
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            if roi_mask is not None:
                grad = grad * roi_mask  # focus or avoid depending on mask semantics you set
            delta = -eps * grad.sign()  # targeted
            delta = apply_frequency(delta, freq)
            delta = delta.clamp(min=-eps, max=eps)
            x_new = (x_orig + delta).clamp(0.0, 1.0)

        params = {
            "method": "FGSM",
            "epsilon": float(eps),
            "target_class": target_class,
            "frequency": freq,
            "tv_weight": float(tv_weight),
        }
        return x_new.detach(), params
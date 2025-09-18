import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import TrackMeta, TrackMetrics
from ...core.utils.images import (
    pil_read_rgb,
    pil_to_tensor_unit,
    tensor_to_pil_unit,
    save_png,
    save_delta_npy,
    pil_to_uint8,
    compute_ssim_rgb_uint8,
    load_roi_mask,
)
from ...core.utils.metrics import tv_norm, fft_magnitude, spectral_energy_bands, spectral_overlap
from ...critique.detectors import (
    load_classifier,
    get_device,
    build_transform_tensor,
)
from ...config import (
    MODELS_DIR,
    DEVICE_CHOICE,
    DIVERSITY_LOW_CUTOFF,
    DIVERSITY_HIGH_CUTOFF,
    TV_WEIGHT_DEFAULT,
    INCLUDE_FINALCHECK_IN_LOSS,
    FINALCHECK_LOSS_WEIGHT,
    FINALCHECK_MODELS_DIR,
    FINALCHECK_NAME,
)

from pathlib import Path
from ...finalcheck.model import load_final_model  # for optional in-loss use


def _merge_strategies(global_s: Optional[dict], advisor_s: Optional[dict]) -> Optional[dict]:
    if not advisor_s and not global_s:
        return None
    global_s = global_s or {}
    advisor_s = advisor_s or {}
    merged = dict(advisor_s)
    merged.update(global_s)
    return merged


def _diff_vit_preprocess(x_unit: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """
    Differentiable preprocessing for ViT: center crop to 224 and normalize.
    Assumes x_unit in [0,1], NCHW, size 256x256.
    """
    N, C, H, W = x_unit.shape
    # Center crop 224
    start_h = (H - out_size) // 2
    start_w = (W - out_size) // 2
    x = x_unit[:, :, start_h:start_h+out_size, start_w:start_w+out_size]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1,3,1,1)
    x = (x - mean) / std
    return x


class MethodAgentBase(ABC):
    def __init__(self, agent_name: str, desk: FileMixingDesk, advisor=None, llm_every_k: int = 5):
        self.agent_name = agent_name
        self.desk = desk
        self.device = get_device(DEVICE_CHOICE)
        self.norm = build_transform_tensor()
        self.advisor = advisor
        self.llm_every_k = max(1, int(llm_every_k))

        models_dir = Path(MODELS_DIR)
        self.models = {
            "resnet50": load_classifier("resnet50", models_dir / "resnet50.pth", self.device),
            "densenet121": load_classifier("densenet121", models_dir / "densenet121.pth", self.device),
        }

        # Optional: load final check model for in-loss usage
        self.final_model = None
        if INCLUDE_FINALCHECK_IN_LOSS:
            fc_weights = Path(FINALCHECK_MODELS_DIR) / f"{FINALCHECK_NAME}.pth"
            if fc_weights.exists():
                self.final_model = load_final_model(FINALCHECK_NAME, fc_weights, self.device)
                self.final_model.eval()
            else:
                print(f"[{self.agent_name}] INCLUDE_FINALCHECK_IN_LOSS=True but weights not found at {fc_weights}")

    @abstractmethod
    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        epsilon_max: float,
        strategy: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        ...

    def _read_strategy(self, image_id: str) -> Optional[dict]:
        doc = self.desk.load_strategy(image_id)
        if doc and isinstance(doc.suggestions, dict):
            return doc.suggestions.get(self.agent_name, {})
        return None

    def _build_loss(self, x_adv: torch.Tensor, target_class: int, tv_weight: float = 0.0, roi_mask: Optional[torch.Tensor] = None):
        """
        Returns callable loss_fn(x_adv) -> scalar tensor combining:
        - CE toward target_class over surrogates
        - Optional CE toward target_class over final model (weighted)
        - Optional TV regularization (weighted)
        - Optional ROI mask weighting (applied to gradients by multiplying grad)
        """
        def loss_fn(x_in: torch.Tensor):
            loss = 0.0
            # Surrogates CE
            for _, m in self.models.items():
                logits = m(self.norm(x_in))
                loss = loss + F.cross_entropy(logits, torch.tensor([target_class], device=x_in.device))
            loss = loss / len(self.models)

            # Optional final model CE
            if self.final_model is not None and FINALCHECK_LOSS_WEIGHT > 0.0:
                x_vit = _diff_vit_preprocess(x_in)
                logits_f = self.final_model(x_vit)
                loss_f = F.cross_entropy(logits_f, torch.tensor([target_class], device=x_in.device))
                loss = loss + FINALCHECK_LOSS_WEIGHT * loss_f

            # Optional TV
            if tv_weight > 0.0:
                loss = loss + tv_weight * tv_norm(x_in)

            return loss
        return loss_fn, roi_mask

    def run_once(self, image_id: str, step: int = 0):
        # Load original (256x256)
        orig_path = self.desk.path_original(image_id)
        img = pil_read_rgb(orig_path, size=None)
        x = pil_to_tensor_unit(img).to(self.device)
        objective = self.desk.load_objective(image_id)

        # Global + per-agent LLM suggestions
        strat = self._read_strategy(image_id)
        advisor_s = None
        if self.advisor is not None and (step % self.llm_every_k == 0):
            try:
                advisor_s = self.advisor.suggest(image_id)
            except Exception:
                advisor_s = None
        merged = _merge_strategies(strat, advisor_s)

        # ROI mask if present (data/mixing_desk/<id>/roi/mask.png)
        roi_path = os.path.join(self.desk._image_dir(image_id), "roi", "mask.png")
        roi = load_roi_mask(roi_path, (x.shape[2], x.shape[3]))
        if roi is not None:
            roi = roi.to(self.device)

        # TV weight from strategy or default
        tv_weight = float(merged.get("tv_weight", TV_WEIGHT_DEFAULT)) if merged else TV_WEIGHT_DEFAULT

        # Generate
        x_adv, params = self.generate(
            x=x.clone(),
            target_class=objective.target_class,
            epsilon_max=objective.epsilon_max,
            strategy=merged,
            tv_weight=tv_weight,
            roi_mask=roi,
        )

        # Metrics
        adv_img = tensor_to_pil_unit(x_adv)
        adv_uint8 = pil_to_uint8(adv_img)
        orig_uint8 = pil_to_uint8(img)
        ssim_val = compute_ssim_rgb_uint8(orig_uint8, adv_uint8)

        # Surrogate target confidence
        def _predict_confs(x_unit: torch.Tensor, target_idx: int) -> Dict[str, float]:
            x_norm = self.norm(x_unit.clone())
            out = {}
            with torch.no_grad():
                for name, m in self.models.items():
                    logits = m(x_norm)
                    probs = F.softmax(logits, dim=1)
                    out[name] = float(probs[0, target_idx].item())
            return out

        if not torch.isfinite(x_adv).all():
            x_adv = torch.nan_to_num(x_adv, nan=0.5, posinf=1.0, neginf=0.0).clamp(0,1)
            
        confs = _predict_confs(x_adv, target_idx=objective.target_class)
        avg_conf = sum(confs.values()) / len(confs)

        # Complementarity metrics on delta
        delta = (x_adv - x).detach()
        mag = fft_magnitude(delta)
        bands = spectral_energy_bands(mag, DIVERSITY_LOW_CUTOFF, DIVERSITY_HIGH_CUTOFF)

        # Pairwise overlap vs one peer (if available)
        overlap = None
        track_names = self.desk.list_tracks(image_id)
        for name in track_names:
            if name == self.agent_name:
                continue
            peer_img_path = os.path.join(self.desk.track_dir(image_id, name), "latest.png")
            if os.path.exists(peer_img_path):
                peer = pil_read_rgb(peer_img_path, size=None)
                peer_t = pil_to_tensor_unit(peer).to(self.device)
                peer_delta = (peer_t - x).detach()
                peer_mag = fft_magnitude(peer_delta)
                overlap = spectral_overlap(mag, peer_mag)
                break

        # Save outputs
        tr_dir = self.desk.track_dir(image_id, self.agent_name)
        img_path = os.path.join(tr_dir, "latest.png")
        delta_path = os.path.join(tr_dir, "latest_delta.npy")
        save_png(adv_img, img_path)
        save_delta_npy(delta.cpu(), delta_path)

        # Extend metrics model
        extra_metrics = {
            "spec_low": bands["low"][0],
            "spec_mid": bands["mid"][0],
            "spec_high": bands["high"][0],
            "spec_overlap": overlap if overlap is not None else None,
            "tv": float(tv_norm(delta).item()),
        }

        meta = TrackMeta(
            agent=self.agent_name,
            method=params.get("method", self.agent_name),
            params=params,
            metrics=TrackMetrics(
                ssim=ssim_val,
                conf_resnet50=confs.get("resnet50"),
                conf_densenet121=confs.get("densenet121"),
                avg_conf=avg_conf,
            ),
            image_path=img_path,
            delta_path=delta_path,
            step=step,
        )
        # Attach extra metrics (non-schema fields go into params to preserve)
        meta.params.update({"metrics_extra": extra_metrics})

        self.desk.write_track(image_id, self.agent_name, meta)

        # Always print params
        param_order = ["epsilon", "alpha", "steps", "frequency", "target_class", "tv_weight"]
        if "tv_weight" not in params:
            params["tv_weight"] = tv_weight
        param_kv = [f"{k}={params[k]}" for k in param_order if k in params]
        if not param_kv:
            param_kv = [f"{k}={v}" for k, v in sorted(params.items())]
        print(
            f"[{self.agent_name}] step={step} params: {', '.join(param_kv)} | "
            f"ssim={ssim_val:.4f} avg_conf={avg_conf:.4f} -> {img_path}"
        )
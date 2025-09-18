import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import MasterMeta, TrackMetrics
from ...core.utils.images import (
    pil_read_rgb,
    pil_to_tensor_unit,
    tensor_to_pil_unit,
    pil_to_uint8,
    compute_ssim_rgb_uint8,
    save_png,
)
from ...critique.detectors import get_device, load_classifier, build_transform_tensor
from ...finalcheck.model import load_final_model, build_spatial_transform_vit
from ...config import (
    MODELS_DIR,
    DEVICE_CHOICE,
    MIXER_MAX_EVALS,
    MIXER_RESTARTS,
    MIXER_USE_FINAL,
    FINALCHECK_MODELS_DIR,
    FINALCHECK_NAME,
    MIXER_SURROGATE_WEIGHT,
    MIXER_FINAL_WEIGHT
)

from pathlib import Path
from PIL import Image

from filelock import FileLock

class MixerAgent:
    def __init__(self, desk: FileMixingDesk):
        self.desk = desk
        self.device = get_device(DEVICE_CHOICE)
        self.norm = build_transform_tensor()
        models_dir = Path(MODELS_DIR)
        self.models = {
            "resnet50": load_classifier("resnet50", models_dir / "resnet50.pth", self.device),
            "densenet121": load_classifier("densenet121", models_dir / "densenet121.pth", self.device),
        }
        self.final_model = None
        self.final_transform = None
        if MIXER_USE_FINAL:
            weights = Path(FINALCHECK_MODELS_DIR) / f"{FINALCHECK_NAME}.pth"
            if weights.exists():
                self.final_model = load_final_model(FINALCHECK_NAME, weights, self.device)
                self.final_model.eval()
                # for evaluation only (no gradient)
                self.final_transform = build_spatial_transform_vit()
            else:
                print(f"[Mixer] MIXER_USE_FINAL=True but final model weights not found at {weights}")

    def _load_track_candidate(self, path: str):
        img = pil_read_rgb(path, size=None)
        x = pil_to_tensor_unit(img)
        return x

    def _blend(self, x0: torch.Tensor, xs: List[torch.Tensor], w: np.ndarray) -> torch.Tensor:
        x_mix = x0.clone()
        for wi, xi in zip(w, xs):
            x_mix = x_mix + float(wi) * (xi - x0)
        return x_mix.clamp(0, 1)

    def _surrogate_conf(self, x_unit: torch.Tensor, target_idx: int) -> Dict[str, float]:
        x_unit = x_unit.to(self.device)  # ensure same device as models
        x_norm = self.norm(x_unit.clone())
        out = {}
        with torch.no_grad():
            for name, m in self.models.items():
                logits = m(x_norm)
                probs = F.softmax(logits, dim=1)
                out[name] = float(probs[0, target_idx].item())
        return out

    def _final_conf(self, x_img: Image.Image, target_idx: int) -> float:
        if self.final_model is None:
            return 0.0
        t = self.final_transform(x_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.final_model(t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            return float(probs[target_idx])

    def _objective(self, x0_img, x_mix_tensor, obj) -> Tuple[float, Dict]:
        # compute ssim and confidences
        x_mix_img = tensor_to_pil_unit(x_mix_tensor)
        ssim_val = compute_ssim_rgb_uint8(pil_to_uint8(x0_img), pil_to_uint8(x_mix_img))
        if ssim_val < obj.ssim_min:
            return -1e9, {"ssim": ssim_val}  # reject

        confs = self._surrogate_conf(x_mix_tensor, obj.target_class)
        avg_conf = sum(confs.values()) / len(confs)
        score = MIXER_SURROGATE_WEIGHT * avg_conf

        final_conf = None
        if self.final_model is not None:
            final_conf = self._final_conf(x_mix_img, obj.target_class)
            score += MIXER_FINAL_WEIGHT * final_conf

        return score, {
            "ssim": ssim_val,
            "conf_resnet50": confs.get("resnet50"),
            "conf_densenet121": confs.get("densenet121"),
            "avg_conf": avg_conf,
            "final_conf": final_conf,
        }

    def _hill_climb(self, x0, xs, obj, start_w: np.ndarray) -> Tuple[np.ndarray, Dict, torch.Tensor]:
        best_w = start_w.copy()
        best_score = -1e9
        best_meta = None
        best_x = None

        for _ in range(MIXER_MAX_EVALS):
            w = best_w + 0.15 * np.random.randn(*best_w.shape)
            w = np.clip(w, 1e-6, None)
            w = w / w.sum()

            x_mix = self._blend(x0, xs, w)
            score, details = self._objective(tensor_to_pil_unit(x0), x_mix, obj)
            if score > best_score:
                best_score = score
                best_w = w
                best_meta = details
                best_x = x_mix

        return best_w, best_meta, best_x

    def run_once(self, image_id: str):
        obj = self.desk.load_objective(image_id)
        x0_img = pil_read_rgb(self.desk.path_original(image_id), size=None)
        x0 = pil_to_tensor_unit(x0_img)

        # Collect tracks
        track_names = self.desk.list_tracks(image_id)
        candidates = {}
        for name in track_names:
            path = os.path.join(self.desk.track_dir(image_id, name), "latest.png")
            if os.path.exists(path):
                candidates[name] = self._load_track_candidate(path)
        if not candidates:
            print("[Mixer] No candidate tracks yet.")
            return

        names = list(candidates.keys())
        xs = [candidates[n] for n in names]

        # Seeds: strategist weights or uniform
        strat = self.desk.load_strategy(image_id)
        if strat and strat.mixer_weights:
            w0 = np.array([float(strat.mixer_weights.get(n, 0.0)) for n in names], dtype=np.float32)
        else:
            w0 = np.ones(len(xs), dtype=np.float32)
        if w0.sum() <= 0:
            w0 = np.ones_like(w0)
        w0 = w0 / w0.sum()

        # Restarts
        best_score = -1e9
        best = None
        best_w = None
        for ri in range(max(1, MIXER_RESTARTS)):
            start_w = w0 if ri == 0 else np.random.dirichlet(np.ones(len(xs))).astype(np.float32)
            bw, meta, x = self._hill_climb(x0, xs, obj, start_w)
            # finalize score check
            score = 0.0
            if meta is not None:
                if meta.get("final_conf") is not None and self.final_model is not None:
                    score = 0.5 * meta["avg_conf"] + 0.5 * meta["final_conf"]
                else:
                    score = meta["avg_conf"]
            if score > best_score:
                best_score = score
                best = meta
                best_w = bw
                best_x = x

        if best is None or best_w is None or best_x is None:
            print("[Mixer] No feasible solution this cycle.")
            return

        # Save master
        mdir = self.desk.master_dir(image_id)
        out_path = os.path.join(mdir, "master.png")
        lock = FileLock(out_path + ".lock")
        with lock:
            save_png(tensor_to_pil_unit(best_x), out_path)
        weights_dict = {n: float(best_w[i]) for i, n in enumerate(names)}
        meta = MasterMeta(
            weights=weights_dict,
            metrics=TrackMetrics(
                ssim=float(best["ssim"]),
                conf_resnet50=best.get("conf_resnet50"),
                conf_densenet121=best.get("conf_densenet121"),
                avg_conf=float(best.get("avg_conf", 0.0)),
            ),
            image_path=out_path,
        )
        self.desk.write_master(image_id, meta)
        print(f"[Mixer] Optimized weights: {weights_dict} | avg_conf={meta.metrics.avg_conf:.4f} ssim={meta.metrics.ssim:.4f} -> {out_path}")
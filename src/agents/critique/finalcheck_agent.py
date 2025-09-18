import os
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from ...core.mixing_desk import FileMixingDesk
from ...core.schemas import CritiqueEntry
from ...critique.detectors import get_device
from ...finalcheck.model import load_final_model, build_spatial_transform_vit
from ...config import (
    FINALCHECK_MODELS_DIR,
    FINALCHECK_NAME,
    FINALCHECK_RESIZE,
    FINALCHECK_CENTER_CROP,
    FINALCHECK_MIN_CONF,
    DEVICE_CHOICE,
)

class FinalCheckAgent:
    def __init__(self, desk: FileMixingDesk):
        self.desk = desk
        self.device = get_device(DEVICE_CHOICE)
        self.name = FINALCHECK_NAME
        weights = Path(FINALCHECK_MODELS_DIR) / f"{self.name}.pth"
        if not weights.exists():
            raise FileNotFoundError(f"FinalCheck weights not found: {weights}")
        self.model = load_final_model(self.name, weights, self.device)
        if self.name == "vit_b_16":
            self.transform = build_spatial_transform_vit(FINALCHECK_RESIZE, FINALCHECK_CENTER_CROP)
        else:
            self.transform = build_spatial_transform_vit(FINALCHECK_RESIZE, FINALCHECK_CENTER_CROP)

    def run_once(self, image_id: str):
        m_path = os.path.join(self.desk.master_dir(image_id), "master.png")
        if not os.path.exists(m_path):
            print(f"[FinalCheck:{self.name}] No master.png yet.")
            return

        obj = self.desk.load_objective(image_id)
        target_idx = int(getattr(obj, "target_class", 0))
        baseline = self.desk.load_baseline(image_id) or {}
        base_pred = (((baseline.get("final") or {}).get(self.name) or {}).get("pred"))

        img = Image.open(m_path).convert("RGB")
        t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            pred = int(logits.argmax(1).item())
            conf_target = float(probs[target_idx])

        success_target = bool(pred == target_idx and conf_target >= FINALCHECK_MIN_CONF)
        flipped = (base_pred is not None) and (pred != int(base_pred))

        entry = CritiqueEntry(
            name=f"FinalCheck_{self.name}",
            kind="final",
            metrics={
                "pred": pred,
                "target_idx": target_idx,
                "conf_target": conf_target,
                "success_target": success_target,
                "flipped": flipped,
                "resize": FINALCHECK_RESIZE,
                "center_crop": FINALCHECK_CENTER_CROP,
                "min_conf": FINALCHECK_MIN_CONF,
            },
            source=m_path,
        )
        self.desk.append_feedback(image_id, entry)
        print(f"[FinalCheck:{self.name}] pred={pred} target={target_idx} conf_target={conf_target:.4f} success_target={success_target} flipped={flipped}")
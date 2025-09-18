"""
verify_final_vit.py

Verify the final master image against the final-check ViT model.

Usage examples:
  - By image_id (looks under data/mixing_desk/<image_id>/master/master.png):
      python verify_final_vit.py --image-id 22adcc46aca5abbb

  - By explicit path to a master image:
      python verify_final_vit.py --master ./data/mixing_desk/22adcc46aca5abbb/master/master.png

Optional:
  --json    Print a JSON object with fields: path, pred, probs, conf
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure local src/ is importable (same directory level as run_attacker.py)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
import torch.nn.functional as F
from PIL import Image

from src.core.mixing_desk import FileMixingDesk
from src.finalcheck.model import load_final_model, build_spatial_transform_vit
from src.critique.detectors import get_device
from src.config import (
    FINALCHECK_MODELS_DIR,
    FINALCHECK_NAME,
    FINALCHECK_RESIZE,
    FINALCHECK_CENTER_CROP,
    DEVICE_CHOICE,
)


def infer_vit(model, transform, img_path: str, device: torch.device):
    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        pred = int(logits.argmax(1).item())
        conf = float(max(probs))  # confidence of predicted class
    return pred, probs, conf


def main():
    p = argparse.ArgumentParser(description="Verify final master image using the ViT final check model.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-id", type=str, help="image_id under data/mixing_desk/<image_id>/master/master.png")
    g.add_argument("--master", type=str, help="Explicit path to master image (PNG/JPG).")
    p.add_argument("--json", action="store_true", help="Output results as JSON.")
    args = p.parse_args()

    # Resolve master image path
    if args.image_id:
        desk = FileMixingDesk()
        master_dir = desk.master_dir(args.image_id)
        master_path = os.path.join(master_dir, "master.png")
        if not os.path.exists(master_path):
            print(f"[ERROR] Master image not found at: {master_path}")
            sys.exit(1)
    else:
        master_path = args.master
        if not os.path.exists(master_path):
            print(f"[ERROR] Master image not found at: {master_path}")
            sys.exit(1)

    # Device and model
    device = get_device(DEVICE_CHOICE)
    weights_path = Path(FINALCHECK_MODELS_DIR) / f"{FINALCHECK_NAME}.pth"
    if not weights_path.exists():
        print(f"[ERROR] Final-check model weights not found: {weights_path}")
        sys.exit(1)

    model = load_final_model(FINALCHECK_NAME, weights_path, device)
    transform = build_spatial_transform_vit(FINALCHECK_RESIZE, FINALCHECK_CENTER_CROP)

    # Inference
    pred, probs, conf = infer_vit(model, transform, master_path, device)

    # Output
    if args.json:
        out = {
            "path": master_path,
            "model": FINALCHECK_NAME,
            "pred": pred,
            "probs": probs,
            "conf": conf,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"[VERIFY] model={FINALCHECK_NAME} path={master_path}")
        print(f"  pred={pred}  conf={conf:.6f}")
        print(f"  probs={probs}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vit_b_16
from pathlib import Path
from typing import Tuple

CLASSES = 2

def build_spatial_transform_vit(resize: int = 256, center_crop: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((resize, resize)),
        T.CenterCrop((center_crop, center_crop)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def load_final_model(name: str, weight_path: Path, device: torch.device) -> nn.Module:
    """
    Currently supports vit_b_16. Extend here for future final-check models.
    """
    if name == "vit_b_16":
        model = vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, CLASSES)
    else:
        raise ValueError(f"Unsupported final check model '{name}'")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model
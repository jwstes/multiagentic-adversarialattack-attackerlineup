import base64
import mimetypes
import os
import time
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import hashlib
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import tempfile

def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))

def image_to_data_url(path: str, force_format: Optional[str] = None) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    if force_format:
        mime = f"image/{force_format.lower()}"
    if mime is None:
        mime = "image/png"
    desired_format = mime.split("/")[-1].upper()
    with Image.open(path) as img:
        buf = BytesIO()
        if img.mode in ("P", "LA"):
            img = img.convert("RGBA")
        elif img.mode == "CMYK":
            img = img.convert("RGB")
        img.save(buf, format=desired_format if desired_format != "JPG" else "JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def image_to_data_url_resized(path: str, max_side: int = 192, fmt: str = "JPEG", quality: int = 70) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        if scale != 1.0:
            im = im.resize((new_w, new_h), Image.BILINEAR)
        buf = BytesIO()
        im.save(buf, format=fmt, quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def pil_open_safe(path: str, tries: int = 8, sleep_sec: float = 0.05) -> Image.Image:
    last_err = None
    for _ in range(tries):
        try:
            img = Image.open(path)
            img.load()  # force read
            return img
        except (UnidentifiedImageError, OSError) as e:
            last_err = e
            time.sleep(sleep_sec)
    raise last_err if last_err else UnidentifiedImageError(f"Cannot open image: {path}")

def pil_read_rgb(path: str, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    img = pil_open_safe(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img

def pil_to_tensor_unit(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t

def tensor_to_pil_unit(x: torch.Tensor) -> Image.Image:
    # x: 1x3xHxW, expected finite in [0,1]
    x = x.detach().cpu()
    # Replace NaN/Inf, then clamp
    x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
    arr = (x.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

def save_png(img: Image.Image, path: str, max_retries: int = 5, sleep: float = 0.05):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # atomic write via temp file + replace with retries (Windows may lock dest)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False, suffix=".tmp") as tmp:
        tmp_path = tmp.name
    try:
        img.save(tmp_path, format="PNG", optimize=True)
        for i in range(max_retries):
            try:
                os.replace(tmp_path, path)  # atomic on same filesystem
                break
            except PermissionError:
                time.sleep(sleep)
        else:
            # Final attempt: try remove and rename
            try:
                if os.path.exists(path):
                    os.remove(path)
                os.replace(tmp_path, path)
            except PermissionError as e:
                raise e
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def compute_ssim_rgb_uint8(a: np.ndarray, b: np.ndarray) -> float:
    return sum(ssim(a[..., c], b[..., c], data_range=255) for c in range(3)) / 3.0

def pil_to_uint8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def save_delta_npy(delta: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np = __import__("numpy")
    np.save(path, delta.detach().cpu().numpy())

def load_roi_mask(path: str, size_hw: tuple[int,int]) -> Optional[torch.Tensor]:
    """
    Load a grayscale mask (H,W) from path, resize to size_hw, return 1x1xH xW float tensor in [0,1].
    White=1 => focus region. Return None if not found.
    """
    if not os.path.exists(path):
        return None
    img = pil_open_safe(path).convert("L").resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    import torch
    t = torch.from_numpy(arr)[None, None, :, :]  # 1x1xH xW
    return t
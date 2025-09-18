import numpy as np
import torch
import torch.nn.functional as F

def tv_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Total variation norm for x (N,C,H,W). Returns scalar tensor.
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

def fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    x: NCHW (delta), returns magnitude spectrum (N, C, H, W) normalized to [0,1] per sample.
    """
    # FFT2 on spatial dims
    X = torch.fft.fft2(x, norm="ortho")
    mag = torch.abs(X)  # NCHW
    # Normalize per-sample
    mag = mag / (mag.amax(dim=(1,2,3), keepdim=True) + 1e-8)
    return mag

def spectral_energy_bands(mag: torch.Tensor, low_cut: float, high_cut: float) -> dict:
    """
    mag: NCHW normalized magnitude
    low_cut/high_cut: fractions of Nyquist (0..0.5 roughly)
    Returns per-sample band energies dict with keys low, mid, high (averaged over C,H,W).
    """
    N, C, H, W = mag.shape
    # Frequency grid [0..0.5] approximately as radius from DC
    fy = torch.linspace(-0.5, 0.5, steps=H, device=mag.device).reshape(1,1,H,1)
    fx = torch.linspace(-0.5, 0.5, steps=W, device=mag.device).reshape(1,1,1,W)
    rad = torch.sqrt(fx**2 + fy**2)  # 1,1,H,W
    low_mask = (rad <= low_cut).float()
    high_mask = (rad >= high_cut).float()
    mid_mask = (1.0 - low_mask) * (1.0 - high_mask)

    def band_energy(mask):
        e = (mag * mask).mean(dim=(1,2,3))  # per-sample
        return e

    low_e = band_energy(low_mask)
    mid_e = band_energy(mid_mask)
    high_e = band_energy(high_mask)

    return {
        "low": low_e.detach().cpu().tolist(),
        "mid": mid_e.detach().cpu().tolist(),
        "high": high_e.detach().cpu().tolist(),
    }

def spectral_overlap(mag_a: torch.Tensor, mag_b: torch.Tensor) -> float:
    """
    Cosine similarity between flattened magnitude spectra (averaged over batch).
    Returns Python float in [0,1].
    """
    a = mag_a.flatten(1)
    b = mag_b.flatten(1)
    num = (a * b).sum(dim=1)
    denom = (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
    cos = (num / denom).mean().item()
    return float(max(0.0, min(1.0, cos)))
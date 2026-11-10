"""Corruptions d'images satellite à intensité contrôlée.

Toutes les fonctions prennent un tenseur (C, H, W) dans [0, 1]
et un niveau de sévérité de 1 à 5, et retournent un tenseur dans [0, 1].
"""
import torch
import torchvision.transforms.functional as TF

LEVELS = (1, 2, 3, 4, 5)

# Paramètres par niveau (à ajuster après observation visuelle)
BLUR_SIGMA   = (0.5, 1.0, 1.5, 2.5, 4.0)
NOISE_STD    = (0.02, 0.05, 0.08, 0.12, 0.18)
HAZE_T       = (0.9, 0.8, 0.7, 0.55, 0.4)    # transmission : plus petit = plus de brume
BRIGHT_DELTA = (0.1, 0.2, 0.3, 0.4, 0.5)


def _check(severity: int) -> int:
    if severity not in LEVELS:
        raise ValueError(f"severity doit être dans {LEVELS}, reçu {severity}")
    return severity - 1


def gaussian_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    sigma = BLUR_SIGMA[_check(severity)]
    k = int(2 * round(3 * sigma) + 1)          # noyau impair
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma]).clamp(0, 1)


def gaussian_noise(x: torch.Tensor, severity: int, seed: int = 0) -> torch.Tensor:
    std = NOISE_STD[_check(severity)]
    g = torch.Generator().manual_seed(seed)
    return (x + std * torch.randn(x.shape, generator=g)).clamp(0, 1)


def haze(x: torch.Tensor, severity: int, airlight: float = 0.85) -> torch.Tensor:
    """Brume atmosphérique : I = J*t + A*(1-t)."""
    t = HAZE_T[_check(severity)]
    return (x * t + airlight * (1 - t)).clamp(0, 1)


def brightness(x: torch.Tensor, severity: int) -> torch.Tensor:
    return (x + BRIGHT_DELTA[_check(severity)]).clamp(0, 1)


CORRUPTIONS = {
    "blur": gaussian_blur,
    "noise": gaussian_noise,
    "haze": haze,
    "brightness": brightness,
}
"""Model construction, losses and checkpoint I/O."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def build_model(arch: str = "Unet", encoder: str = "resnet34", weights: str | None = "imagenet",
                in_channels: int = 1) -> torch.nn.Module:
    """in_channels inputs (1 = grayscale H-alpha, 3 = image + ridge + top-hat), single logit output.

    smp adapts the pretrained first conv to 1 channel by summing RGB filters, and uses the
    RGB filters directly for 3 channels.
    """
    return smp.create_model(arch, encoder_name=encoder, encoder_weights=weights,
                            in_channels=in_channels, classes=1)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0) -> torch.Tensor:
    """BCE (works with soft targets) + soft Dice over the batch."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice_weight * dice

_LOVASZ = None


def bce_lovasz_loss(logits: torch.Tensor, target: torch.Tensor, lovasz_weight: float = 1.0) -> torch.Tensor:
    """BCE on the soft (annotator-averaged) target + Lovasz hinge on the majority mask.

    The Lovasz hinge is a convex surrogate of the Jaccard index (IoU), computed per image,
    so it directly pushes masks across the IoU > 0.5 bar that decides a PQ match.
    It needs a binary target, so pixels marked by at least half of the annotators count.
    """
    global _LOVASZ
    if _LOVASZ is None:
        _LOVASZ = smp.losses.LovaszLoss(mode="binary", per_image=True, from_logits=True)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + lovasz_weight * _LOVASZ(logits, (target >= 0.5).float())

def save_checkpoint(path, model, cfg, half: bool = False, **extra):
    sd = model.state_dict()
    if half:
        sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
    torch.save({"state_dict": sd, "config": cfg, **extra}, path)


def load_checkpoint(path, device="cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = build_model(mcfg["arch"], mcfg["encoder"], weights=None, in_channels=mcfg.get("in_channels", 1))
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), ckpt["config"]

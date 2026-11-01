"""
utils.py — Shared utilities for the monocular depth estimation project.

Covers:
  - Config loading (YAML)
  - Reproducibility setup (seeds, deterministic mode)
  - Depth colorization for visualization
  - Checkpoint save/load helpers
  - A simple AverageMeter for tracking running metrics
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for servers and Colab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image


# ─────────────────────────────────────────────────────────────
# 1. Config Loading
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a nested dictionary.

    We use a plain dict (not an argparse Namespace or dataclass) so that
    config values can be accessed with cfg['model']['encoder'] — explicit
    and easy to understand for beginners.

    Args:
        config_path: Path to the .yaml config file.

    Returns:
        Nested dictionary of config values.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────
# 2. Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Why multiple seeds? Python's random, NumPy, and PyTorch each maintain
    their own RNG state. Setting all three ensures that every shuffle,
    initialization, and dropout mask is identical across runs.

    Args:
        seed:          Integer seed value (e.g., 42).
        deterministic: If True, forces CUDA to use deterministic algorithms.
                       This can slow down training ~5–10% but guarantees
                       bit-exact reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # for multi-GPU setups

    if deterministic:
        # Forces CUDA to pick deterministic algorithms where available.
        # Some ops (e.g., atomicAdd in scatter) have no deterministic
        # implementation — torch will raise an error if they're used.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required env var for deterministic scatter/gather ops
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # benchmark=True lets cuDNN auto-select the fastest conv algorithm
        # for your input sizes. Great for fixed-size inputs in production.
        torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────
# 3. Depth Map Colorization
# ─────────────────────────────────────────────────────────────

def colorize_depth(
    depth: np.ndarray,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    colormap: str = "magma",
) -> np.ndarray:
    """
    Convert a single-channel float depth map to an RGB colorized image.

    The depth values are normalized to [0, 1] then mapped through a
    matplotlib colormap. This is purely for visualization — the colorized
    image is NOT used in training.

    Args:
        depth:     2D numpy array of shape (H, W) with float depth values.
        min_depth: Minimum depth for normalization. If None, uses array min.
        max_depth: Maximum depth for normalization. If None, uses array max.
        colormap:  Matplotlib colormap name. "magma" and "turbo" look best
                   for depth visualization.

    Returns:
        RGB image as uint8 numpy array of shape (H, W, 3).
    """
    # Replace NaN/Inf with 0 before normalizing (invalid pixels)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = min_depth if min_depth is not None else depth.min()
    vmax = max_depth if max_depth is not None else depth.max()

    # Avoid division by zero if depth map is constant (e.g., all zeros)
    if vmax == vmin:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)

    # Apply colormap: returns RGBA float array in [0, 1]
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap)
    colorized = cmap(normalized)[:, :, :3]   # drop alpha channel

    # Convert to uint8 for saving / display
    colorized = (colorized * 255).astype(np.uint8)
    return colorized


# ─────────────────────────────────────────────────────────────
# 4. Visualization: 4-Panel Plot
# ─────────────────────────────────────────────────────────────

def visualize_batch(
    images: torch.Tensor,
    gt_depths: torch.Tensor,
    pred_depths: torch.Tensor,
    masks: torch.Tensor,
    n_samples: int = 4,
    colormap: str = "magma",
    min_depth: float = 0.001,
    max_depth: float = 10.0,
) -> np.ndarray:
    """
    Create a 4-column visualization grid:
        Col 1: Input RGB image
        Col 2: Ground truth depth (colorized)
        Col 3: Predicted depth (colorized)
        Col 4: Absolute error map (valid pixels only)

    Args:
        images:     Batch of RGB images, shape (B, 3, H, W), values in [0, 1].
        gt_depths:  Batch of GT depth maps, shape (B, 1, H, W).
        pred_depths: Batch of predicted depth maps, shape (B, 1, H, W).
        masks:      Validity masks, shape (B, 1, H, W), bool.
        n_samples:  Number of samples to show (rows in the grid).
        colormap:   Matplotlib colormap for depth visualization.
        min_depth:  Depth range minimum for consistent colorization.
        max_depth:  Depth range maximum for consistent colorization.

    Returns:
        Grid image as uint8 numpy array of shape (H_grid, W_grid, 3).
    """
    n = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    # Handle the case where n=1 (axes is 1D, not 2D)
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input RGB", "Ground Truth Depth", "Predicted Depth", "Abs Error"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        # ── RGB image ──────────────────────────────────────
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[i, 0].imshow(img_np)

        # ── Ground truth depth ─────────────────────────────
        gt_np = gt_depths[i, 0].cpu().numpy()
        gt_color = colorize_depth(gt_np, min_depth, max_depth, colormap)
        axes[i, 1].imshow(gt_color)

        # ── Predicted depth ────────────────────────────────
        pred_np = pred_depths[i, 0].detach().cpu().numpy()
        pred_color = colorize_depth(pred_np, min_depth, max_depth, colormap)
        axes[i, 2].imshow(pred_color)

        # ── Absolute error (only at valid pixels) ──────────
        mask_np = masks[i, 0].cpu().numpy().astype(bool)
        error = np.zeros_like(gt_np)
        error[mask_np] = np.abs(pred_np[mask_np] - gt_np[mask_np])
        # Normalize error to [0, 1] relative to max error in this sample
        max_err = error.max() if error.max() > 0 else 1.0
        error_color = colorize_depth(error, 0.0, max_err, "hot")
        axes[i, 3].imshow(error_color)

        # Remove axis ticks for cleaner look
        for col in range(4):
            axes[i, col].axis("off")

    plt.tight_layout()

    # Render figure to numpy array without saving to disk
    fig.canvas.draw()
    try:
        # Matplotlib >= 3.8
        grid_np = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        # Older Matplotlib fallback
        grid_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid_np = grid_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return grid_np


# ─────────────────────────────────────────────────────────────
# 5. Checkpoint Utilities
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
) -> None:
    """
    Save a training checkpoint to disk.

    The 'state' dict typically contains:
        - 'epoch': current epoch number
        - 'model_state_dict': model.state_dict()
        - 'optimizer_state_dict': optimizer.state_dict()
        - 'scheduler_state_dict': scheduler.state_dict()
        - 'scaler_state_dict': GradScaler.state_dict() (for AMP)
        - 'best_metric': best validation metric so far
        - 'cfg': the config dict (so experiments are self-contained)

    Args:
        state:           Dictionary of objects to save.
        checkpoint_dir:  Directory to save the checkpoint in.
        filename:        Filename (e.g., "best_model.pth").
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"  [✓] Checkpoint saved → {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        filepath:   Path to the .pth checkpoint file.
        model:      The model to load weights into.
        optimizer:  Optional optimizer to restore state.
        scheduler:  Optional LR scheduler to restore state.
        device:     Device to map checkpoint tensors to.

    Returns:
        The full checkpoint dictionary (contains epoch, best_metric, etc.).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # map_location ensures checkpoints saved on GPU load correctly on CPU
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  [✓] Model weights loaded from epoch {checkpoint.get('epoch', '?')}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# ─────────────────────────────────────────────────────────────
# 6. AverageMeter — Running Mean Tracker
# ─────────────────────────────────────────────────────────────

class AverageMeter:
    """
    Tracks the running average of a scalar (e.g., loss, metric).

    Usage:
        meter = AverageMeter("loss")
        meter.update(0.42, n=8)     # n = batch size
        print(meter.avg)            # → running average
        meter.reset()               # reset for next epoch
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Args:
            val: Value to add (e.g., batch loss).
            n:   Weight of this value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: val={self.val:.4f}, avg={self.avg:.4f})"


# ─────────────────────────────────────────────────────────────
# 7. Sanity Checks (run these during development)
# ─────────────────────────────────────────────────────────────

def check_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    expected_shape: Optional[tuple] = None,
) -> None:
    """
    Quick sanity check: print tensor stats and optionally verify shape.

    Call this liberally during development to catch NaN/Inf early,
    which are the silent killers of depth estimation training.

    Args:
        tensor:         Tensor to inspect.
        name:           Descriptive name for the print statement.
        expected_shape: If provided, assert that tensor.shape == expected_shape.
    """
    print(f"\n{'─'*40}")
    print(f"[check] {name}")
    print(f"  shape  : {tuple(tensor.shape)}")
    print(f"  dtype  : {tensor.dtype}")
    print(f"  device : {tensor.device}")
    print(f"  min    : {tensor.min().item():.4f}")
    print(f"  max    : {tensor.max().item():.4f}")
    print(f"  mean   : {tensor.mean().item():.4f}")
    print(f"  has_nan: {torch.isnan(tensor).any().item()}")
    print(f"  has_inf: {torch.isinf(tensor).any().item()}")
    print(f"{'─'*40}\n")

    if expected_shape is not None:
        assert tuple(tensor.shape) == tuple(expected_shape), (
            f"Shape mismatch for '{name}': "
            f"expected {expected_shape}, got {tuple(tensor.shape)}"
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Useful for comparing architectures. Print this before training starts.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters (as an integer).
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [i] Trainable parameters: {total:,}")
    return total

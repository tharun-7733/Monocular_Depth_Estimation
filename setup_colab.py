"""
setup_colab.py — One-shot environment setup for Google Colab.

Run this script ONCE at the start of a new Colab session.
It installs all dependencies, creates the folder structure,
and verifies GPU availability.

Usage (in a Colab cell):
    !python setup_colab.py
    
Or paste each section into separate Colab cells.
"""

# ─────────────────────────────────────────────────────────────
# CELL 1: Install dependencies
# ─────────────────────────────────────────────────────────────
# Run this first. It takes ~2 minutes on first run.

import subprocess
import sys

print("📦 Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
])
print("✅ Dependencies installed.")


# ─────────────────────────────────────────────────────────────
# CELL 2: Verify GPU
# ─────────────────────────────────────────────────────────────

import torch

print("\n🔍 Environment Check")
print(f"  Python version : {sys.version.split()[0]}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
    # Get VRAM in GB
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  VRAM           : {vram_gb:.1f} GB")

    if vram_gb < 8:
        print("\n⚠️  WARNING: Less than 8 GB VRAM detected.")
        print("   Recommendation: Set batch_size=4 and input_height=192, input_width=256")
        print("   in config/default.yaml")
    elif vram_gb < 12:
        print("\n✅ VRAM OK. Default config (batch_size=8, 320x256) should work.")
    else:
        print("\n🚀 Plenty of VRAM. You can try batch_size=16 or input size 480x640.")
else:
    print("\n❌ No GPU found. Training will be extremely slow on CPU.")
    print("   In Colab: Runtime → Change runtime type → T4 GPU")


# ─────────────────────────────────────────────────────────────
# CELL 3: Create project folder structure
# ─────────────────────────────────────────────────────────────

from pathlib import Path

dirs = [
    "config",
    "data/nyu_depth_v2",
    "checkpoints",
    "runs",
    "outputs",
]

print("\n📁 Creating project directories...")
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {d}/")

print("\n✅ Project structure ready.")


# ─────────────────────────────────────────────────────────────
# CELL 4: Quick smoke test — verify core imports work
# ─────────────────────────────────────────────────────────────

print("\n🔬 Smoke-testing imports...")

try:
    import numpy as np
    print("   ✓ numpy")

    import yaml
    print("   ✓ yaml")

    import PIL
    print("   ✓ Pillow")

    import cv2
    print("   ✓ OpenCV")

    import matplotlib
    print("   ✓ matplotlib")

    from torch.utils.tensorboard import SummaryWriter
    print("   ✓ TensorBoard")

    import torchvision
    print("   ✓ torchvision")

    print("\n✅ All imports OK — you're ready to proceed!")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("   Try re-running CELL 1 (pip install).")


# ─────────────────────────────────────────────────────────────
# CELL 5: Enable TensorBoard in Colab
# ─────────────────────────────────────────────────────────────
# In Colab, run this magic command in a separate cell:
#
#   %load_ext tensorboard
#   %tensorboard --logdir runs
#
# This opens a live TensorBoard panel inside the Colab notebook.
# You'll see loss curves appear after training starts.

print("\n📊 TensorBoard Instructions for Colab:")
print("   Add a new cell and run:")
print("     %load_ext tensorboard")
print("     %tensorboard --logdir runs")
print("\n   Then run train.py in the next cell.")
print("   TensorBoard updates in real time — refresh to see latest curves.")

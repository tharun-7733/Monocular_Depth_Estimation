"""
app.py — FastAPI backend for Monocular Depth Estimation using trained_depth_model.

Model Priority:
  1. Local trained model: ./trained_depth_model (Depth Anything ViT-S SOTA model)
  2. Fallback: Intel MiDaS via torch.hub
  3. Fallback: DPT via HuggingFace
  4. Fallback: ResNet-18 depth proxy
"""

import os
import sys
import time
import io
import base64
import warnings
from pathlib import Path
from typing import Optional

# Monkeypatch numpy compatibility for SciPy in environments with NumPy 1.26.x
import numpy as np
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

import torch
import torch.nn as nn
from PIL import Image
import cv2

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore")

# ── Device selection ──────────────────────────────────────────
def get_optimal_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_optimal_device()
BASE_DIR = Path(__file__).resolve().parent
TRAINED_MODEL_DIR = BASE_DIR / "trained_depth_model"
FRONTEND_DIR = BASE_DIR / "frontend"

# ── App Definition ────────────────────────────────────────────
app = FastAPI(
    title="Depth Estimation API",
    description="Real-time monocular depth recognition powered by local Depth Anything model",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model State ───────────────────────────────────────────────
_model = None
_processor = None
_model_kind = None
_model_name = "Not loaded"

def load_trained_local_model() -> bool:
    global _model, _processor, _model_kind, _model_name
    if not TRAINED_MODEL_DIR.exists():
        print(f"[!] {TRAINED_MODEL_DIR} does not exist.")
        return False
    
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f"[i] Loading local trained depth model from {TRAINED_MODEL_DIR} on {DEVICE}...")
        processor = AutoImageProcessor.from_pretrained(str(TRAINED_MODEL_DIR))
        model = AutoModelForDepthEstimation.from_pretrained(str(TRAINED_MODEL_DIR)).to(DEVICE)
        model.eval()
        
        _model = model
        _processor = processor
        _model_kind = "depth_anything_local"
        _model_name = "Depth Anything (Local trained_depth_model)"
        print(f"[✓] Successfully loaded local trained model: {_model_name}")
        return True
    except Exception as e:
        print(f"[!] Error loading local trained_depth_model: {e}")
        return False

def try_load_midas() -> bool:
    global _model, _processor, _model_kind, _model_name
    try:
        os.environ["TORCH_HUB_SKIP_GITHUB_VALIDATION"] = "1"
        m = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, skip_validation=True).to(DEVICE)
        m.eval()
        _model = m
        _processor = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)
        _model_kind = "midas"
        _model_name = "MiDaS_small (torch.hub fallback)"
        print(f"[✓] Loaded {_model_name}")
        return True
    except Exception as e:
        print(f"[!] MiDaS fallback failed: {e}")
        return False

class ResNetDepthProxy(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.fuse = nn.Sequential(
            nn.Conv2d(960, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        s = self.stem(x)
        f1 = self.layer1(s)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        up = lambda f: nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        fused = torch.cat([up(f1), up(f2), up(f3), up(f4)], dim=1)
        return self.fuse(fused)

def load_proxy() -> bool:
    global _model, _processor, _model_kind, _model_name
    try:
        m = ResNetDepthProxy().to(DEVICE).eval()
        _model = m
        _processor = None
        _model_kind = "proxy"
        _model_name = "ResNet-18 Depth Proxy (fallback)"
        print(f"[✓] Loaded {_model_name}")
        return True
    except Exception as e:
        print(f"[!] ResNet proxy fallback failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    print("[*] Initializing Depth Estimation Engine...")
    success = load_trained_local_model() or try_load_midas() or load_proxy()
    if not success:
        print("[X] Failed to load any depth model!")
    else:
        print(f"[*] Engine ready. Active Model: {_model_name} on {DEVICE}")

# ── Inference Engine ──────────────────────────────────────────
def run_depth_estimation(img_pil: Image.Image) -> np.ndarray:
    """
    Runs depth estimation on a PIL image and returns a float32 depth map of size (H, W).
    """
    W, H = img_pil.size
    
    if _model_kind == "depth_anything_local":
        inputs = {k: v.to(DEVICE) for k, v in _processor(images=img_pil, return_tensors="pt").items()}
        with torch.no_grad():
            outputs = _model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Interpolate predicted depth to original image dimensions
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy().astype(np.float32)
            return depth_map

    elif _model_kind == "midas":
        img_np = np.array(img_pil)
        batch = _processor.small_transform(img_np).to(DEVICE)
        with torch.no_grad():
            pred = _model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            return pred.cpu().numpy().astype(np.float32)

    else: # proxy
        import torchvision.transforms as T
        preprocess = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = _model(tensor)
            out_resized = torch.nn.functional.interpolate(
                out, size=(H, W), mode="bilinear", align_corners=False
            ).squeeze()
            return out_resized.cpu().numpy().astype(np.float32)

# ── Colormap and Visualization Helpers ────────────────────────
COLORMAPS = {
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "twilight": cv2.COLORMAP_TWILIGHT,
    "ocean": cv2.COLORMAP_OCEAN,
    "gray": None,
}

def normalize_depth_8bit(depth: np.ndarray, invert: bool = True) -> np.ndarray:
    d = depth.copy()
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < 1e-8:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    
    # Invert so closer objects are brighter or vice versa
    if invert:
        norm = 255 - norm
    return norm

def colorize_depth(depth_8bit: np.ndarray, colormap_name: str = "magma") -> np.ndarray:
    cmap = COLORMAPS.get(colormap_name.lower(), cv2.COLORMAP_MAGMA)
    if cmap is None:
        return np.stack([depth_8bit] * 3, axis=-1)
    bgr = cv2.applyColorMap(depth_8bit, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def compute_edge_map(depth_8bit: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth_8bit, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_8bit, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    m_min, m_max = float(mag.min()), float(mag.max())
    if m_max - m_min > 1e-8:
        mag = ((mag - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
    else:
        mag = np.zeros_like(mag, dtype=np.uint8)
    bgr = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def arr_to_base64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── API Endpoints ─────────────────────────────────────────────
@app.get("/")
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Depth Estimation API is running. Mount frontend to view UI."})

@app.get("/health")
async def health():
    return {
        "ready": _model is not None,
        "model": _model_name,
        "model_kind": _model_kind,
        "device": str(DEVICE),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "model_folder_present": TRAINED_MODEL_DIR.exists(),
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    colormap: str = Form(default="magma"),
    invert_depth: bool = Form(default=False),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Depth model is not yet loaded.")
    
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
    
    t0 = time.perf_counter()
    depth_raw = run_depth_estimation(pil_img)
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    orig_np = np.array(pil_img)
    depth_8u = normalize_depth_8bit(depth_raw, invert=invert_depth)
    colorized = colorize_depth(depth_8u, colormap_name=colormap)
    edges = compute_edge_map(depth_8u)
    
    # Also prepare a downsampled 64x64 or 128x128 depth grid for instant 3D interactive viewer
    h, w = depth_raw.shape
    preview_scale = min(128.0 / max(h, w), 1.0)
    new_w, new_h = max(int(w * preview_scale), 16), max(int(h * preview_scale), 16)
    downsampled = cv2.resize(depth_8u, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return JSONResponse({
        "success": True,
        "model_used": _model_name,
        "device": str(DEVICE),
        "inference_ms": inference_ms,
        "image_size": {"w": pil_img.size[0], "h": pil_img.size[1]},
        "colormap": colormap,
        "invert_depth": invert_depth,
        "depth_stats": {
            "min": round(float(depth_raw.min()), 3),
            "mean": round(float(depth_raw.mean()), 3),
            "max": round(float(depth_raw.max()), 3),
            "std": round(float(depth_raw.std()), 3),
        },
        "original": arr_to_base64_png(orig_np),
        "colorized_depth": arr_to_base64_png(colorized),
        "edge_map": arr_to_base64_png(edges),
        "raw_depth_grayscale": arr_to_base64_png(depth_8u),
        "depth_grid_preview": downsampled.tolist(),
        "depth_grid_dims": {"w": new_w, "h": new_h},
    })

# Mount frontend directory for assets/static
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

"""
preprocessing.py
----------------
Part 1 — Pre-processing
  • BGR → YCbCr color space conversion
  • 4:2:0 chroma subsampling (Cb, Cr downsampled by 2× in both dimensions)
"""

import cv2
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
YCbCrFrame = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (Y, Cb, Cr)


# ---------------------------------------------------------------------------
# Forward pass: BGR → YCbCr with 4:2:0 chroma subsampling
# ---------------------------------------------------------------------------

def bgr_to_ycbcr(bgr_frame: np.ndarray) -> YCbCrFrame:
    """
    Convert a BGR image (uint8, H×W×3) to YCbCr and apply 4:2:0 subsampling.

    Returns
    -------
    Y  : (H, W)       luma channel, float32, range ≈ [0, 255]
    Cb : (H//2, W//2) chroma-blue, float32, range ≈ [0, 255]
    Cr : (H//2, W//2) chroma-red,  float32, range ≈ [0, 255]
    """
    # OpenCV uses the BT.601 formula; result is still uint8 [0,255]
    ycbcr = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)  # note: OpenCV orders Y, Cr, Cb
    Y  = ycbcr[:, :, 0].astype(np.float32)
    Cr = ycbcr[:, :, 1].astype(np.float32)
    Cb = ycbcr[:, :, 2].astype(np.float32)

    # 4:2:0 — downsample Cb and Cr by 2× in each dimension (area average)
    Cb_sub = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Cr_sub = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2), interpolation=cv2.INTER_AREA)

    return Y, Cb_sub, Cr_sub


# ---------------------------------------------------------------------------
# Inverse pass: YCbCr (4:2:0) → BGR
# ---------------------------------------------------------------------------

def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Upsample Cb/Cr back to full resolution and convert to BGR uint8.

    Parameters
    ----------
    Y  : (H, W)       float32 luma
    Cb : (H//2, W//2) float32 chroma-blue (subsampled)
    Cr : (H//2, W//2) float32 chroma-red  (subsampled)

    Returns
    -------
    bgr : (H, W, 3) uint8
    """
    h, w = Y.shape

    # Upsample chroma channels back to full resolution
    Cb_up = cv2.resize(Cb, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Stack in OpenCV's YCrCb order and convert back
    ycrcb = np.stack([
        np.clip(Y,     0, 255).astype(np.uint8),
        np.clip(Cr_up, 0, 255).astype(np.uint8),
        np.clip(Cb_up, 0, 255).astype(np.uint8),
    ], axis=-1)

    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr


# ---------------------------------------------------------------------------
# Utility: load a folder of frames
# ---------------------------------------------------------------------------

def load_frames(frames_dir: str) -> list[np.ndarray]:
    """Load all PNG/JPG frames from a directory, sorted by filename."""
    import os
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(
        f for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    frames = []
    for fname in files:
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Cannot read frame: {path}")
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    Y, Cb, Cr = bgr_to_ycbcr(img)
    print(f"Original : {img.shape}")
    print(f"Y  shape : {Y.shape}  | min={Y.min():.1f} max={Y.max():.1f}")
    print(f"Cb shape : {Cb.shape} | min={Cb.min():.1f} max={Cb.max():.1f}")
    print(f"Cr shape : {Cr.shape} | min={Cr.min():.1f} max={Cr.max():.1f}")

    reconstructed = ycbcr_to_bgr(Y, Cb, Cr)
    cv2.imwrite("reconstructed_preview.png", reconstructed)
    print("Reconstructed frame saved → reconstructed_preview.png")
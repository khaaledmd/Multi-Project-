"""
intra_coding.py
---------------
Part 2 — Intra-frame Coding (I-frames)
  • Block-based 8×8 DCT
  • Quantisation with a configurable quantisation matrix
  • Inverse path: de-quantisation + IDCT → reconstructed channel
"""

import numpy as np
from scipy.fft import dctn, idctn
from typing import Tuple


# ---------------------------------------------------------------------------
# Standard JPEG-like luminance quantisation matrix (quality ≈ 50)
# ---------------------------------------------------------------------------
LUMA_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float32)

CHROMA_QUANT_MATRIX = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def get_quant_matrix(channel: str = "luma", quality: float = 50.0) -> np.ndarray:
    """
    Return a scaled 8×8 quantisation matrix.

    quality : 1 (worst) … 100 (lossless-ish)
    """
    base = LUMA_QUANT_MATRIX if channel == "luma" else CHROMA_QUANT_MATRIX

    # Standard JPEG quality scaling
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    Q = np.floor((base * scale + 50) / 100).astype(np.float32)
    Q = np.clip(Q, 1, 255)
    return Q


# ---------------------------------------------------------------------------
# Block-level DCT helpers
# ---------------------------------------------------------------------------

def dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II on an 8×8 block."""
    return dctn(block, type=2, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    """2-D inverse DCT-II on an 8×8 block."""
    return idctn(block, type=2, norm="ortho")


# ---------------------------------------------------------------------------
# Channel-level encode / decode
# ---------------------------------------------------------------------------

def encode_channel(
    channel: np.ndarray,
    Q: np.ndarray,
    block_size: int = 8,
) -> np.ndarray:
    """
    Encode a single 2-D channel (Y, Cb, or Cr) using block DCT + quantisation.

    The channel is padded so its dimensions are multiples of block_size.

    Returns
    -------
    coeffs : int16 array of the same (padded) shape, containing quantised DCT coefficients.
    """
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")

    ph, pw = padded.shape
    coeffs = np.zeros((ph, pw), dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size] - 128.0  # level shift
            dct_block = dct2(block)
            q_block = np.round(dct_block / Q)
            coeffs[i:i+block_size, j:j+block_size] = q_block

    return coeffs.astype(np.int16)


def decode_channel(
    coeffs: np.ndarray,
    Q: np.ndarray,
    original_shape: Tuple[int, int],
    block_size: int = 8,
) -> np.ndarray:
    """
    Decode quantised DCT coefficients back to a spatial channel.

    Parameters
    ----------
    coeffs         : int16 array (padded shape)
    Q              : quantisation matrix
    original_shape : (H, W) of the unpadded channel

    Returns
    -------
    channel : float32 array of shape original_shape, values in [0, 255]
    """
    ph, pw = coeffs.shape
    reconstructed = np.zeros((ph, pw), dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            q_block = coeffs[i:i+block_size, j:j+block_size].astype(np.float32)
            dct_block = q_block * Q          # de-quantise
            block = idct2(dct_block) + 128.0 # IDCT + undo level shift
            reconstructed[i:i+block_size, j:j+block_size] = block

    # Crop to original size and clamp
    h, w = original_shape
    reconstructed = reconstructed[:h, :w]
    return np.clip(reconstructed, 0, 255)


# ---------------------------------------------------------------------------
# I-frame encode / decode (all three channels)
# ---------------------------------------------------------------------------

def encode_iframe(
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    quality: float = 50.0,
) -> dict:
    """
    Encode an I-frame.

    Returns a dict with:
        'Y_coeffs'  : int16 ndarray
        'Cb_coeffs' : int16 ndarray
        'Cr_coeffs' : int16 ndarray
        'Y_shape'   : (H, W)
        'Cb_shape'  : (H//2, W//2)
        'Cr_shape'  : (H//2, W//2)
        'quality'   : float
    """
    Ql = get_quant_matrix("luma",   quality)
    Qc = get_quant_matrix("chroma", quality)

    return {
        "frame_type": "I",
        "Y_coeffs":   encode_channel(Y,  Ql),
        "Cb_coeffs":  encode_channel(Cb, Qc),
        "Cr_coeffs":  encode_channel(Cr, Qc),
        "Y_shape":    Y.shape,
        "Cb_shape":   Cb.shape,
        "Cr_shape":   Cr.shape,
        "quality":    quality,
    }


def decode_iframe(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode an I-frame dict back to (Y, Cb, Cr) float32 channels.
    """
    quality = data["quality"]
    Ql = get_quant_matrix("luma",   quality)
    Qc = get_quant_matrix("chroma", quality)

    Y  = decode_channel(data["Y_coeffs"],  Ql, data["Y_shape"])
    Cb = decode_channel(data["Cb_coeffs"], Qc, data["Cb_shape"])
    Cr = decode_channel(data["Cr_coeffs"], Qc, data["Cr_shape"])

    return Y, Cb, Cr


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    import sys
    from preprocessing import bgr_to_ycbcr, ycbcr_to_bgr

    if len(sys.argv) < 2:
        print("Usage: python intra_coding.py <image_path> [quality]")
        sys.exit(1)

    quality = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0

    img = cv2.imread(sys.argv[1])
    Y, Cb, Cr = bgr_to_ycbcr(img)

    data = encode_iframe(Y, Cb, Cr, quality=quality)
    Y_r, Cb_r, Cr_r = decode_iframe(data)
    out = ycbcr_to_bgr(Y_r, Cb_r, Cr_r)

    cv2.imwrite("iframe_reconstructed.png", out)

    # PSNR
    mse = np.mean((img.astype(np.float64) - out.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
    print(f"Quality={quality} | PSNR={psnr:.2f} dB")
    print("Saved → iframe_reconstructed.png")
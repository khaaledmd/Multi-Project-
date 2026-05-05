"""
inter_coding.py
---------------
Part 3 — Inter-frame Coding (P-frames)
  • Group of Pictures (GOP): every G-th frame → I-frame, rest → P-frames
  • Block matching (full search) on the Y channel with ±S pixel search window
  • Motion vectors stored per 16×16 macroblock
  • Residual = current − prediction → DCT + quantisation
  • Decoder: IDCT → de-quantise residual + motion-compensated prediction
"""

import numpy as np
from typing import Tuple, List
from inter_coding import (
    encode_iframe, decode_iframe,
    encode_channel, decode_channel,
    get_quant_matrix,
    dct2, idct2,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MB_SIZE = 16      # macroblock size (pixels)
BLOCK_SIZE = 8    # DCT block size


# ---------------------------------------------------------------------------
# Motion estimation helpers
# ---------------------------------------------------------------------------

def _block_sad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Sum of Absolute Differences between two equal-sized blocks."""
    return np.sum(np.abs(block_a.astype(np.float32) - block_b.astype(np.float32)))


def motion_estimate_macroblock(
    current_mb: np.ndarray,
    ref_frame: np.ndarray,
    mb_row: int,
    mb_col: int,
    search_window: int,
) -> Tuple[int, int]:
    """
    Full-search block matching for one 16×16 macroblock.

    Parameters
    ----------
    current_mb    : (MB_SIZE, MB_SIZE) float32 — current macroblock (Y channel)
    ref_frame     : (H, W) float32 — full reference (previous reconstructed) Y channel
    mb_row, mb_col: top-left corner of the macroblock in the frame
    search_window : ±S pixels search range

    Returns
    -------
    (dy, dx) : best motion vector (row-offset, col-offset)
    """
    H, W = ref_frame.shape
    best_sad = float("inf")
    best_dy, best_dx = 0, 0

    for dy in range(-search_window, search_window + 1):
        for dx in range(-search_window, search_window + 1):
            ref_row = mb_row + dy
            ref_col = mb_col + dx

            # Boundary check
            if ref_row < 0 or ref_col < 0:
                continue
            if ref_row + MB_SIZE > H or ref_col + MB_SIZE > W:
                continue

            ref_mb = ref_frame[ref_row:ref_row+MB_SIZE, ref_col:ref_col+MB_SIZE]
            sad = _block_sad(current_mb, ref_mb)

            if sad < best_sad:
                best_sad = sad
                best_dy, best_dx = dy, dx

    return best_dy, best_dx


# ---------------------------------------------------------------------------
# Residual encoding / decoding (DCT on 8×8 sub-blocks of the residual)
# ---------------------------------------------------------------------------

def _encode_residual_channel(residual: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """DCT + quantise a residual channel (same shape as input)."""
    h, w = residual.shape
    pad_h = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    padded = np.pad(residual, ((0, pad_h), (0, pad_w)), mode="constant")
    ph, pw = padded.shape

    coeffs = np.zeros((ph, pw), dtype=np.float32)
    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            block = padded[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            dct_block = dct2(block)
            coeffs[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = np.round(dct_block / Q)

    return coeffs[:h, :w].astype(np.int16)


def _decode_residual_channel(coeffs: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """De-quantise + IDCT a residual channel."""
    h, w = coeffs.shape
    pad_h = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    padded = np.pad(coeffs.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="constant")
    ph, pw = padded.shape

    out = np.zeros((ph, pw), dtype=np.float32)
    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            q_block = padded[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            out[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = idct2(q_block * Q)

    return out[:h, :w]


# ---------------------------------------------------------------------------
# P-frame encode / decode
# ---------------------------------------------------------------------------

def encode_pframe(
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    ref_Y: np.ndarray,
    ref_Cb: np.ndarray,
    ref_Cr: np.ndarray,
    search_window: int = 8,
    quality: float = 50.0,
) -> dict:
    """
    Encode a P-frame given the current and reference (previous reconstructed) channels.

    Returns a dict with:
        'frame_type'  : 'P'
        'motion_vecs' : list of (mb_row, mb_col, dy, dx)
        'Y_residual'  : int16 ndarray
        'Cb_residual' : int16 ndarray (on subsampled chroma)
        'Cr_residual' : int16 ndarray
        'Y_shape'     : (H, W)
        'Cb_shape'    : (H//2, W//2)
        'quality'     : float
    """
    H, W = Y.shape
    Ql = get_quant_matrix("luma",   quality)
    Qc = get_quant_matrix("chroma", quality)

    # --- Motion estimation on Y channel ---
    motion_vecs: List[Tuple[int, int, int, int]] = []
    Y_pred = np.zeros_like(Y)

    for mb_row in range(0, H - MB_SIZE + 1, MB_SIZE):
        for mb_col in range(0, W - MB_SIZE + 1, MB_SIZE):
            current_mb = Y[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE]
            dy, dx = motion_estimate_macroblock(current_mb, ref_Y, mb_row, mb_col, search_window)
            motion_vecs.append((mb_row, mb_col, dy, dx))

            # Build luma prediction from reference
            ref_mb = ref_Y[mb_row+dy:mb_row+dy+MB_SIZE, mb_col+dx:mb_col+dx+MB_SIZE]
            Y_pred[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE] = ref_mb

    # Handle right/bottom edge macroblocks (partial)
    # — copy from reference with zero motion for simplicity
    if H % MB_SIZE != 0:
        Y_pred[-(H % MB_SIZE):, :] = ref_Y[-(H % MB_SIZE):, :]
    if W % MB_SIZE != 0:
        Y_pred[:, -(W % MB_SIZE):] = ref_Y[:, -(W % MB_SIZE):]

    # --- Luma residual ---
    Y_residual_spatial = Y - Y_pred
    Y_residual_coeffs  = _encode_residual_channel(Y_residual_spatial, Ql)

    # --- Chroma: use nearest-neighbour motion (scaled ÷2) for Cb/Cr ---
    Cb_pred = _chroma_predict(ref_Cb, motion_vecs, Cb.shape)
    Cr_pred = _chroma_predict(ref_Cr, motion_vecs, Cr.shape)

    Cb_residual = _encode_residual_channel(Cb - Cb_pred, Qc)
    Cr_residual = _encode_residual_channel(Cr - Cr_pred, Qc)

    return {
        "frame_type":  "P",
        "motion_vecs": motion_vecs,
        "Y_residual":  Y_residual_coeffs,
        "Cb_residual": Cb_residual,
        "Cr_residual": Cr_residual,
        "Y_pred":      Y_pred,      # kept for evaluation; can be dropped to save memory
        "Y_shape":     Y.shape,
        "Cb_shape":    Cb.shape,
        "Cr_shape":    Cr.shape,
        "quality":     quality,
    }


def decode_pframe(
    data: dict,
    ref_Y: np.ndarray,
    ref_Cb: np.ndarray,
    ref_Cr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode a P-frame dict back to (Y, Cb, Cr) float32 channels.
    """
    quality = data["quality"]
    Ql = get_quant_matrix("luma",   quality)
    Qc = get_quant_matrix("chroma", quality)
    motion_vecs = data["motion_vecs"]

    # --- Reconstruct luma prediction from motion vectors ---
    H, W = data["Y_shape"]
    Y_pred = np.zeros((H, W), dtype=np.float32)

    for (mb_row, mb_col, dy, dx) in motion_vecs:
        ref_mb = ref_Y[mb_row+dy:mb_row+dy+MB_SIZE, mb_col+dx:mb_col+dx+MB_SIZE]
        Y_pred[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE] = ref_mb

    # Edge fill
    if H % MB_SIZE != 0:
        Y_pred[-(H % MB_SIZE):, :] = ref_Y[-(H % MB_SIZE):, :]
    if W % MB_SIZE != 0:
        Y_pred[:, -(W % MB_SIZE):] = ref_Y[:, -(W % MB_SIZE):]

    # --- Decode residual and add to prediction ---
    Y_residual = _decode_residual_channel(data["Y_residual"], Ql)
    Y = np.clip(Y_pred + Y_residual, 0, 255)

    # --- Chroma ---
    Cb_pred = _chroma_predict(ref_Cb, motion_vecs, data["Cb_shape"])
    Cr_pred = _chroma_predict(ref_Cr, motion_vecs, data["Cr_shape"])

    Cb_residual = _decode_residual_channel(data["Cb_residual"], Qc)
    Cr_residual = _decode_residual_channel(data["Cr_residual"], Qc)

    Cb = np.clip(Cb_pred + Cb_residual, 0, 255)
    Cr = np.clip(Cr_pred + Cr_residual, 0, 255)

    return Y, Cb, Cr


# ---------------------------------------------------------------------------
# Chroma prediction helper (apply ÷2 scaled motion vectors to subsampled chroma)
# ---------------------------------------------------------------------------

def _chroma_predict(
    ref_chroma: np.ndarray,
    motion_vecs: List[Tuple[int, int, int, int]],
    chroma_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Build chroma prediction from luma motion vectors (scaled by ½ for 4:2:0).
    """
    cH, cW = chroma_shape
    pred = np.zeros((cH, cW), dtype=np.float32)
    MB_C = MB_SIZE // 2  # macroblock size in chroma domain (8×8)

    for (mb_row, mb_col, dy, dx) in motion_vecs:
        # Scale coordinates to chroma domain
        cr = mb_row // 2
        cc = mb_col // 2
        cry = dy // 2
        crx = dx // 2

        r0, c0 = cr + cry, cc + crx
        r1, c1 = cr + MB_C, cc + MB_C

        # Boundary checks
        if r0 < 0 or c0 < 0 or r0 + MB_C > cH or c0 + MB_C > cW:
            continue
        if r1 > cH or c1 > cW:
            continue

        pred[cr:r1, cc:c1] = ref_chroma[r0:r0+MB_C, c0:c0+MB_C]

    return pred


# ---------------------------------------------------------------------------
# Full GOP encoder
# ---------------------------------------------------------------------------

def encode_gop(
    frames_ycbcr: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gop_size: int = 10,
    search_window: int = 8,
    quality: float = 50.0,
) -> List[dict]:
    """
    Encode a list of (Y, Cb, Cr) frames into a list of frame dicts (I or P).

    Parameters
    ----------
    frames_ycbcr : list of (Y, Cb, Cr) tuples
    gop_size     : every gop_size-th frame is an I-frame (index 0, G, 2G, …)
    search_window: ±S pixel search range for motion estimation
    quality      : quantisation quality factor (1–100)

    Returns
    -------
    encoded_frames : list of dicts, each with 'frame_type' == 'I' or 'P'
    """
    encoded_frames: List[dict] = []
    ref_Y = ref_Cb = ref_Cr = None   # reference (previous reconstructed) frame

    for idx, (Y, Cb, Cr) in enumerate(frames_ycbcr):
        is_iframe = (idx % gop_size == 0)

        if is_iframe or ref_Y is None:
            # ---- I-frame ----
            data = encode_iframe(Y, Cb, Cr, quality=quality)
            # Decode immediately to get reference for next P-frames
            ref_Y, ref_Cb, ref_Cr = decode_iframe(data)
        else:
            # ---- P-frame ----
            data = encode_pframe(
                Y, Cb, Cr,
                ref_Y, ref_Cb, ref_Cr,
                search_window=search_window,
                quality=quality,
            )
            ref_Y, ref_Cb, ref_Cr = decode_pframe(data, ref_Y, ref_Cb, ref_Cr)

        data["frame_index"] = idx
        encoded_frames.append(data)
        print(f"  Frame {idx:03d} → {data['frame_type']}-frame encoded")

    return encoded_frames


def decode_gop(encoded_frames: List[dict]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Decode a list of encoded frame dicts back to (Y, Cb, Cr) tuples.
    """
    decoded: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    ref_Y = ref_Cb = ref_Cr = None

    for data in encoded_frames:
        if data["frame_type"] == "I":
            Y, Cb, Cr = decode_iframe(data)
        else:
            Y, Cb, Cr = decode_pframe(data, ref_Y, ref_Cb, ref_Cr)

        ref_Y, ref_Cb, ref_Cr = Y, Cb, Cr
        decoded.append((Y, Cb, Cr))

    return decoded


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    import sys
    from preprocessing import bgr_to_ycbcr, ycbcr_to_bgr, load_frames

    if len(sys.argv) < 2:
        print("Usage: python inter_coding.py <frames_dir> [gop_size] [search_window] [quality]")
        sys.exit(1)

    frames_dir    = sys.argv[1]
    gop_size      = int(sys.argv[2])   if len(sys.argv) > 2 else 5
    search_window = int(sys.argv[3])   if len(sys.argv) > 3 else 8
    quality       = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0

    print(f"Loading frames from {frames_dir} …")
    bgr_frames = load_frames(frames_dir)[:20]  # limit to 20 for quick test
    ycbcr_frames = [bgr_to_ycbcr(f) for f in bgr_frames]

    print(f"\nEncoding {len(ycbcr_frames)} frames | GOP={gop_size} | S={search_window} | Q={quality}")
    encoded = encode_gop(ycbcr_frames, gop_size=gop_size, search_window=search_window, quality=quality)

    print("\nDecoding …")
    decoded = decode_gop(encoded)

    # PSNR on last frame
    Y_r, Cb_r, Cr_r = decoded[-1]
    out = ycbcr_to_bgr(Y_r, Cb_r, Cr_r)
    orig = bgr_frames[-1]

    mse = np.mean((orig.astype(np.float64) - out.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
    print(f"\nLast frame PSNR: {psnr:.2f} dB")

    cv2.imwrite("pframe_reconstructed.png", out)
    print("Saved → pframe_reconstructed.png")
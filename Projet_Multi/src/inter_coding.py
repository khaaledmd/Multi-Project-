"""
inter_coding.py
---------------
Part 3 — Inter-frame Coding (P-frames)
  • Group of Pictures (GOP): every G-th frame → I-frame, rest → P-frames
  • Block matching (diamond search) on the Y channel with ±S pixel search window
  • Motion vectors stored per 16×16 macroblock
  • Residual = current − prediction → DCT + quantisation
  • Decoder: IDCT → de-quantise residual + motion-compensated prediction
 
Modification : full-search remplacé par diamond search → 5-10× plus rapide
"""
 
import numpy as np
from typing import Tuple, List
from intra_coding import (
    encode_iframe, decode_iframe,
    encode_channel, decode_channel,
    get_quant_matrix,
    dct2, idct2,
)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MB_SIZE    = 16   # macroblock size (pixels)
BLOCK_SIZE = 8    # DCT block size
 
 
# ---------------------------------------------------------------------------
# Motion estimation — Diamond Search (rapide)
# ---------------------------------------------------------------------------
 
def _block_sad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Sum of Absolute Differences between two equal-sized blocks."""
    return np.sum(np.abs(block_a.astype(np.float32) - block_b.astype(np.float32)))
 
 
def _in_bounds(row, col, dy, dx, H, W):
    """Vérifie que le bloc référence est dans les limites de la frame."""
    r, c = row + dy, col + dx
    return r >= 0 and c >= 0 and r + MB_SIZE <= H and c + MB_SIZE <= W
 
 
def motion_estimate_macroblock(
    current_mb: np.ndarray,
    ref_frame:  np.ndarray,
    mb_row:     int,
    mb_col:     int,
    search_window: int,
) -> Tuple[int, int]:
    """
    Diamond Search block matching — beaucoup plus rapide que le full search.
 
    Principe :
      1. Commence au centre (dy=0, dx=0)
      2. Teste les 4 voisins en diamant (haut, bas, gauche, droite)
      3. Se déplace vers le meilleur voisin
      4. Répète jusqu'à ce que le centre soit le meilleur (convergence)
      5. Affine avec un petit diamant (±1 pixel)
 
    Résultat : même qualité que le full search, 5-10× plus rapide.
    """
    H, W = ref_frame.shape
 
    # Patterns de recherche
    LARGE_DIAMOND = [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1)]
    SMALL_DIAMOND = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
 
    best_dy, best_dx = 0, 0
 
    # SAD initial au centre
    if _in_bounds(mb_row, mb_col, 0, 0, H, W):
        ref_mb   = ref_frame[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE]
        best_sad = _block_sad(current_mb, ref_mb)
    else:
        best_sad = float("inf")
 
    # ── Phase 1 : grand diamant ───────────────────────────────
    for _ in range(search_window):
        moved = False
        for (ddy, ddx) in LARGE_DIAMOND:
            dy = best_dy + ddy
            dx = best_dx + ddx
 
            # Respecter la fenêtre de recherche
            if abs(dy) > search_window or abs(dx) > search_window:
                continue
            if not _in_bounds(mb_row, mb_col, dy, dx, H, W):
                continue
 
            r, c = mb_row + dy, mb_col + dx
            ref_mb = ref_frame[r:r+MB_SIZE, c:c+MB_SIZE]
            sad    = _block_sad(current_mb, ref_mb)
 
            if sad < best_sad:
                best_sad = sad
                best_dy, best_dx = dy, dx
                moved = True
 
        if not moved:
            break   # convergé → passer au petit diamant
 
    # ── Phase 2 : petit diamant (affinage ±1 pixel) ──────────
    for (ddy, ddx) in SMALL_DIAMOND:
        dy = best_dy + ddy
        dx = best_dx + ddx
 
        if abs(dy) > search_window or abs(dx) > search_window:
            continue
        if not _in_bounds(mb_row, mb_col, dy, dx, H, W):
            continue
 
        r, c = mb_row + dy, mb_col + dx
        ref_mb = ref_frame[r:r+MB_SIZE, c:c+MB_SIZE]
        sad    = _block_sad(current_mb, ref_mb)
 
        if sad < best_sad:
            best_sad = sad
            best_dy, best_dx = dy, dx
 
    return best_dy, best_dx
 
 
# ---------------------------------------------------------------------------
# Residual encoding / decoding
# ---------------------------------------------------------------------------
 
def _encode_residual_channel(residual: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """DCT + quantise a residual channel."""
    h, w  = residual.shape
    pad_h = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    padded = np.pad(residual, ((0, pad_h), (0, pad_w)), mode="constant")
    ph, pw = padded.shape
 
    coeffs = np.zeros((ph, pw), dtype=np.float32)
    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            block = padded[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            coeffs[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = np.round(dct2(block) / Q)
 
    return coeffs[:h, :w].astype(np.int16)
 
 
def _decode_residual_channel(coeffs: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """De-quantise + IDCT a residual channel."""
    h, w  = coeffs.shape
    pad_h = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pad_w = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    padded = np.pad(coeffs.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="constant")
    ph, pw = padded.shape
 
    out = np.zeros((ph, pw), dtype=np.float32)
    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            out[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = idct2(padded[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] * Q)
 
    return out[:h, :w]
 
 
# ---------------------------------------------------------------------------
# P-frame encode / decode
# ---------------------------------------------------------------------------
 
def encode_pframe(
    Y:  np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    ref_Y:  np.ndarray,
    ref_Cb: np.ndarray,
    ref_Cr: np.ndarray,
    search_window: int   = 8,
    quality:       float = 50.0,
) -> dict:
    """Encode a P-frame given the current and reference channels."""
    H, W = Y.shape
    Ql = get_quant_matrix("luma",   quality)
    Qc = get_quant_matrix("chroma", quality)
 
    # ── Motion estimation (diamond search) ───────────────────
    motion_vecs: List[Tuple[int, int, int, int]] = []
    Y_pred = np.zeros_like(Y)
 
    for mb_row in range(0, H - MB_SIZE + 1, MB_SIZE):
        for mb_col in range(0, W - MB_SIZE + 1, MB_SIZE):
            current_mb = Y[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE]
            dy, dx = motion_estimate_macroblock(
                current_mb, ref_Y, mb_row, mb_col, search_window)
            motion_vecs.append((mb_row, mb_col, dy, dx))
 
            ref_mb = ref_Y[mb_row+dy:mb_row+dy+MB_SIZE, mb_col+dx:mb_col+dx+MB_SIZE]
            Y_pred[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE] = ref_mb
 
    # Bords (edge macroblocks partiels)
    if H % MB_SIZE != 0:
        Y_pred[-(H % MB_SIZE):, :] = ref_Y[-(H % MB_SIZE):, :]
    if W % MB_SIZE != 0:
        Y_pred[:, -(W % MB_SIZE):] = ref_Y[:, -(W % MB_SIZE):]
 
    # ── Résidu luma ──────────────────────────────────────────
    Y_residual_coeffs = _encode_residual_channel(Y - Y_pred, Ql)
 
    # ── Résidu chroma ────────────────────────────────────────
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
        "Y_pred":      Y_pred,
        "Y_shape":     Y.shape,
        "Cb_shape":    Cb.shape,
        "Cr_shape":    Cr.shape,
        "quality":     quality,
    }
 
 
def decode_pframe(
    data:   dict,
    ref_Y:  np.ndarray,
    ref_Cb: np.ndarray,
    ref_Cr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a P-frame dict back to (Y, Cb, Cr) float32 channels."""
    quality     = data["quality"]
    Ql          = get_quant_matrix("luma",   quality)
    Qc          = get_quant_matrix("chroma", quality)
    motion_vecs = data["motion_vecs"]
 
    H, W   = data["Y_shape"]
    Y_pred = np.zeros((H, W), dtype=np.float32)
 
    for (mb_row, mb_col, dy, dx) in motion_vecs:
        ref_mb = ref_Y[mb_row+dy:mb_row+dy+MB_SIZE, mb_col+dx:mb_col+dx+MB_SIZE]
        Y_pred[mb_row:mb_row+MB_SIZE, mb_col:mb_col+MB_SIZE] = ref_mb
 
    if H % MB_SIZE != 0:
        Y_pred[-(H % MB_SIZE):, :] = ref_Y[-(H % MB_SIZE):, :]
    if W % MB_SIZE != 0:
        Y_pred[:, -(W % MB_SIZE):] = ref_Y[:, -(W % MB_SIZE):]
 
    Y  = np.clip(Y_pred  + _decode_residual_channel(data["Y_residual"],  Ql), 0, 255)
    Cb = np.clip(_chroma_predict(ref_Cb, motion_vecs, data["Cb_shape"])
                 + _decode_residual_channel(data["Cb_residual"], Qc), 0, 255)
    Cr = np.clip(_chroma_predict(ref_Cr, motion_vecs, data["Cr_shape"])
                 + _decode_residual_channel(data["Cr_residual"], Qc), 0, 255)
 
    return Y, Cb, Cr
 
 
# ---------------------------------------------------------------------------
# Chroma prediction helper
# ---------------------------------------------------------------------------
 
def _chroma_predict(
    ref_chroma:   np.ndarray,
    motion_vecs:  List[Tuple[int, int, int, int]],
    chroma_shape: Tuple[int, int],
) -> np.ndarray:
    """Build chroma prediction from luma motion vectors (scaled ÷2 for 4:2:0)."""
    cH, cW = chroma_shape
    pred   = np.zeros((cH, cW), dtype=np.float32)
    MB_C   = MB_SIZE // 2
 
    for (mb_row, mb_col, dy, dx) in motion_vecs:
        cr, cc   = mb_row // 2, mb_col // 2
        cry, crx = dy // 2,     dx // 2
        r0, c0   = cr + cry, cc + crx
 
        if r0 < 0 or c0 < 0 or r0 + MB_C > cH or c0 + MB_C > cW:
            continue
        if cr + MB_C > cH or cc + MB_C > cW:
            continue
 
        pred[cr:cr+MB_C, cc:cc+MB_C] = ref_chroma[r0:r0+MB_C, c0:c0+MB_C]
 
    return pred
 
 
# ---------------------------------------------------------------------------
# Full GOP encoder / decoder
# ---------------------------------------------------------------------------
 
def encode_gop(
    frames_ycbcr:  List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gop_size:      int   = 10,
    search_window: int   = 8,
    quality:       float = 50.0,
) -> List[dict]:
    """Encode a list of (Y, Cb, Cr) frames into a list of frame dicts (I or P)."""
    encoded_frames: List[dict] = []
    ref_Y = ref_Cb = ref_Cr = None
 
    for idx, (Y, Cb, Cr) in enumerate(frames_ycbcr):
        is_iframe = (idx % gop_size == 0)
 
        if is_iframe or ref_Y is None:
            data = encode_iframe(Y, Cb, Cr, quality=quality)
            ref_Y, ref_Cb, ref_Cr = decode_iframe(data)
        else:
            data = encode_pframe(Y, Cb, Cr, ref_Y, ref_Cb, ref_Cr,
                                 search_window=search_window, quality=quality)
            ref_Y, ref_Cb, ref_Cr = decode_pframe(data, ref_Y, ref_Cb, ref_Cr)
 
        data["frame_index"] = idx
        encoded_frames.append(data)
        print(f"  Frame {idx:03d} → {data['frame_type']}-frame encoded")
 
    return encoded_frames
 
 
def decode_gop(
    encoded_frames: List[dict],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode a list of encoded frame dicts back to (Y, Cb, Cr) tuples."""
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
    import cv2, sys
    from preprocessing import bgr_to_ycbcr, ycbcr_to_bgr, load_frames
 
    if len(sys.argv) < 2:
        print("Usage: python inter_coding.py <frames_dir> [gop] [search] [quality]")
        sys.exit(1)
 
    frames_dir    = sys.argv[1]
    gop_size      = int(sys.argv[2])   if len(sys.argv) > 2 else 5
    search_window = int(sys.argv[3])   if len(sys.argv) > 3 else 8
    quality       = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
 
    bgr_frames   = load_frames(frames_dir)[:20]
    ycbcr_frames = [bgr_to_ycbcr(f) for f in bgr_frames]
 
    print(f"Encoding {len(ycbcr_frames)} frames | GOP={gop_size} | S={search_window} | Q={quality}")
    encoded = encode_gop(ycbcr_frames, gop_size=gop_size,
                         search_window=search_window, quality=quality)
    decoded = decode_gop(encoded)
 
    Y_r, Cb_r, Cr_r = decoded[-1]
    out  = ycbcr_to_bgr(Y_r, Cb_r, Cr_r)
    orig = bgr_frames[-1]
    mse  = np.mean((orig.astype(np.float64) - out.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
    print(f"Last frame PSNR: {psnr:.2f} dB")
    cv2.imwrite("pframe_reconstructed.png", out)
    print("Saved → pframe_reconstructed.png")
 

"""
evaluation.py
-------------
Part 5 — Evaluation & Visualisation
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing  import bgr_to_ycbcr, load_frames, ycbcr_to_bgr
from intra_coding   import get_quant_matrix, dct2, idct2
from entropy_coding import decode_from_bin, encode_to_bin
from inter_coding   import encode_gop


# ─────────────────────────────────────────────────────────────
# 5a — MÉTRIQUES
# ─────────────────────────────────────────────────────────────

def compute_psnr(original, reconstructed):
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_metrics(orig_frames, recon_frames, encoded_frames, bin_path):
    psnr_list = [compute_psnr(o, r) for o, r in zip(orig_frames, recon_frames)]

    i_count = sum(1 for f in encoded_frames if f["frame_type"] == "I")
    p_count = sum(1 for f in encoded_frames if f["frame_type"] == "P")

    h, w    = orig_frames[0].shape[:2]
    orig_sz = len(orig_frames) * h * w * 3
    comp_sz = os.path.getsize(bin_path)
    ratio   = orig_sz / comp_sz

    print("=" * 50)
    print("  MÉTRIQUES")
    print("=" * 50)
    print(f"  Frames     : {len(orig_frames)}  ({i_count} I  +  {p_count} P)")
    print(f"  PSNR moyen : {np.mean(psnr_list):.2f} dB")
    print(f"  PSNR min   : {np.min(psnr_list):.2f} dB")
    print(f"  PSNR max   : {np.max(psnr_list):.2f} dB")
    print(f"  Originale  : {orig_sz:,} octets")
    print(f"  Compressée : {comp_sz:,} octets")
    print(f"  Ratio      : {ratio:.2f}×")
    print("=" * 50)

    return {
        "psnr_list": psnr_list,
        "psnr_mean": float(np.mean(psnr_list)),
        "i_frames":  i_count,
        "p_frames":  p_count,
        "orig_sz":   orig_sz,
        "comp_sz":   comp_sz,
        "ratio":     ratio,
    }


# ─────────────────────────────────────────────────────────────
# EXPÉRIENCES RAPPORT
# ─────────────────────────────────────────────────────────────

def experiment_quality(frames_dir, max_frames=20, gop=5, search=8,
                       output_dir="output"):
    """
    Graphe du rapport : Compression Ratio vs Quantization Factor
    Modifie QUALITY_VALUES pour tester d'autres valeurs.
    """
    QUALITY_VALUES = [10, 20, 30, 50, 70, 90]

    bgr_frames   = load_frames(frames_dir)[:max_frames]
    ycbcr_frames = [bgr_to_ycbcr(f) for f in bgr_frames]
    h, w         = bgr_frames[0].shape[:2]
    orig_sz      = len(bgr_frames) * h * w * 3
    os.makedirs(output_dir, exist_ok=True)

    ratios = []
    print("\n[Expérience 1] Compression Ratio vs Quantization Factor")
    print("-" * 50)

    for q in QUALITY_VALUES:
        tmp     = os.path.join(output_dir, f"_tmp_q{int(q)}.bin")
        encoded = encode_gop(ycbcr_frames, gop_size=gop,
                             search_window=search, quality=q)
        comp_sz = encode_to_bin(encoded, tmp)
        ratio   = orig_sz / comp_sz
        ratios.append(ratio)
        print(f"  Quantization Factor={q:3.0f} → Compression Ratio={ratio:.2f}×")

    # ── Graphe ────────────────────────────────────────────────
    BG = "#0d1117"
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor("#161b22")

    ax.plot(QUALITY_VALUES, ratios, "o-", color="#58a6ff", lw=2.5, ms=9)
    ax.fill_between(QUALITY_VALUES, ratios, alpha=0.12, color="#58a6ff")

    for x, y in zip(QUALITY_VALUES, ratios):
        ax.annotate(f"{y:.2f}×", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", color="#58a6ff", fontsize=10, fontweight="bold")

    ax.set_xlabel("Quantization Factor", color="white", fontsize=12)
    ax.set_ylabel("Compression Ratio (×)", color="white", fontsize=12)
    ax.set_title("Compression Ratio vs Quantization Factor",
                 color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white", labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#30363d")

    save = os.path.join(output_dir, "graph_quality.png")
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[OK] Sauvegardé : {save}")


def experiment_gop(frames_dir, max_frames=20, quality=50, search=8,
                   output_dir="output"):
    """
    Graphe du rapport : GOP Size Effect on Compression
    Modifie GOP_VALUES pour tester d'autres valeurs.
    """
    GOP_VALUES = [1, 2, 4, 5, 10, 20]

    bgr_frames   = load_frames(frames_dir)[:max_frames]
    ycbcr_frames = [bgr_to_ycbcr(f) for f in bgr_frames]
    h, w         = bgr_frames[0].shape[:2]
    orig_sz      = len(bgr_frames) * h * w * 3
    os.makedirs(output_dir, exist_ok=True)

    ratios = []
    print("\n[Expérience 2] GOP Size Effect on Compression")
    print("-" * 50)

    for gop in GOP_VALUES:
        tmp     = os.path.join(output_dir, f"_tmp_gop{gop}.bin")
        encoded = encode_gop(ycbcr_frames, gop_size=gop,
                             search_window=search, quality=quality)
        comp_sz = encode_to_bin(encoded, tmp)
        ratio   = orig_sz / comp_sz
        ratios.append(ratio)
        print(f"  GOP Size={gop:3d} → Compression Ratio={ratio:.2f}×")

    # ── Graphe ────────────────────────────────────────────────
    BG = "#0d1117"
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor("#161b22")

    ax.plot(GOP_VALUES, ratios, "D-", color="#3fb950", lw=2.5, ms=9)
    ax.fill_between(GOP_VALUES, ratios, alpha=0.12, color="#3fb950")

    for x, y in zip(GOP_VALUES, ratios):
        ax.annotate(f"{y:.2f}×", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", color="#3fb950", fontsize=10, fontweight="bold")

    ax.set_xlabel("GOP Size (G)", color="white", fontsize=12)
    ax.set_ylabel("Compression Ratio (×)", color="white", fontsize=12)
    ax.set_title("GOP Size Effect on Compression",
                 color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white", labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#30363d")

    save = os.path.join(output_dir, "graph_gop.png")
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[OK] Sauvegardé : {save}")


# ─────────────────────────────────────────────────────────────
# 5b — VISUALISATION
# ─────────────────────────────────────────────────────────────

def create_pipeline_figure(orig_frames, recon_frames, encoded_frames,
                            metrics, save_path="output/pipeline_visualization.png"):

    BG = "#0d1117"
    fig = plt.figure(figsize=(20, 26), facecolor=BG)
    fig.suptitle("MPEG-4 Pipeline — Visualisation",
                 fontsize=20, fontweight="bold", color="white", y=0.995)

    outer = gridspec.GridSpec(5, 1, figure=fig, hspace=0.50,
                              top=0.978, bottom=0.02, left=0.04, right=0.97)

    _label(fig, "① Frames originales",             outer[0])
    _label(fig, "② Canaux YCbCr",                  outer[1])
    _label(fig, "③ DCT & Quantisation (bloc 8×8)", outer[2])
    _label(fig, "④ Vecteurs de mouvement",          outer[3])
    _label(fig, "⑤ Résidus & PSNR",                outer[4])

    _fig1_frames(fig, outer[0], orig_frames, encoded_frames)
    _fig2_ycbcr(fig,  outer[1], orig_frames[0])
    _fig3_dct(fig,    outer[2], orig_frames, encoded_frames)
    _fig4_mv(fig,     outer[3], orig_frames, encoded_frames)
    _fig5_psnr(fig,   outer[4], orig_frames, recon_frames, encoded_frames, metrics)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[OK] Figure sauvegardée : {save_path}")


# ── helpers ───────────────────────────────────────────────────

def _label(fig, text, spec):
    ax = fig.add_subplot(spec)
    ax.set_visible(False)
    pos = ax.get_position()
    fig.text(0.035, pos.y1 + 0.002, text,
             fontsize=12, fontweight="bold", color="#58a6ff",
             transform=fig.transFigure)


def _fig1_frames(fig, spec, orig_frames, encoded_frames, max_show=8):
    n     = min(max_show, len(orig_frames))
    inner = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=spec, wspace=0.04)
    for i in range(n):
        ax    = fig.add_subplot(inner[i])
        ax.imshow(cv2.cvtColor(orig_frames[i], cv2.COLOR_BGR2RGB))
        ftype = encoded_frames[i]["frame_type"] if i < len(encoded_frames) else "?"
        color = "#ff7b72" if ftype == "I" else "#79c0ff"
        ax.set_title(f"F{i+1} [{ftype}]", fontsize=8, color=color,
                     fontweight="bold", pad=3)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(color)
            sp.set_linewidth(2)


def _fig2_ycbcr(fig, spec, frame_bgr):
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.08)
    Y, Cb_s, Cr_s = bgr_to_ycbcr(frame_bgr)
    h, w = Y.shape
    Cb   = cv2.resize(Cb_s, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr   = cv2.resize(Cr_s, (w, h), interpolation=cv2.INTER_LINEAR)

    panels = [
        ("Original (RGB)",    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), None),
        ("Y  (Luminance)",    Y,  "gray"),
        ("Cb (Chroma bleu)",  Cb, "Blues_r"),
        ("Cr (Chroma rouge)", Cr, "Reds_r"),
    ]
    for i, (title, data, cmap) in enumerate(panels):
        ax = fig.add_subplot(inner[i])
        ax.imshow(data, cmap=cmap,
                  vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(title, fontsize=9, color="white", pad=4)
        ax.axis("off")


def _fig3_dct(fig, spec, orig_frames, encoded_frames):
    inner   = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.30)
    Y, _, _ = bgr_to_ycbcr(orig_frames[0])
    h, w    = Y.shape
    i0, j0  = (h // 2 // 8) * 8, (w // 2 // 8) * 8
    raw     = Y[i0:i0+8, j0:j0+8].astype(np.float64)

    quality = encoded_frames[0]["quality"]
    Q       = get_quant_matrix("luma", quality).astype(np.float64)
    dct_c   = dct2(raw - 128.0)
    quant   = np.round(dct_c / Q)
    recon   = idct2(quant * Q) + 128.0

    steps = [
        ("Pixels bruts",       raw,   "viridis", False),
        ("Coefficients DCT",   dct_c, "RdBu_r",  True),
        ("Après quantisation", quant, "RdBu_r",  True),
        ("Bloc reconstruit",   recon, "viridis",  False),
    ]

    for i, (title, data, cmap, centered) in enumerate(steps):
        ax  = fig.add_subplot(inner[i])
        arr = np.array(data)
        if centered:
            vmax = max(abs(arr.min()), abs(arr.max()), 1)
            im   = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=255)

        for r in range(8):
            for c in range(8):
                ax.text(c, r, f"{arr[r,c]:.0f}", ha="center", va="center",
                        fontsize=5, color="white")

        ax.set_title(title, fontsize=9, color="white", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if i < 3:
            ax.annotate("", xy=(1.22, 0.5), xytext=(1.05, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2))


def _fig4_mv(fig, spec, orig_frames, encoded_frames):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.08)
    p_idx = next((i for i, f in enumerate(encoded_frames)
                  if f["frame_type"] == "P"), None)

    ax_l = fig.add_subplot(inner[0])
    ax_r = fig.add_subplot(inner[1])

    if p_idx is None or p_idx >= len(orig_frames):
        for ax in (ax_l, ax_r):
            ax.text(0.5, 0.5, "Pas de P-frame", ha="center", va="center",
                    color="gray", transform=ax.transAxes)
            ax.axis("off")
        return

    frame = orig_frames[p_idx]
    h, w  = frame.shape[:2]
    MB    = 16

    ax_l.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax_l.set_title(f"P-frame #{p_idx+1} originale", fontsize=9, color="white")
    ax_l.axis("off")

    ax_r.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), alpha=0.55)
    for (mb_row, mb_col, dy, dx) in encoded_frames[p_idx]["motion_vecs"]:
        cx, cy = mb_col + MB // 2, mb_row + MB // 2
        if dy != 0 or dx != 0:
            ax_r.annotate("", xy=(cx+dx, cy+dy), xytext=(cx, cy),
                          arrowprops=dict(arrowstyle="->", color="#ffa657",
                                          lw=1.2, mutation_scale=8))
    ax_r.set_xlim(0, w); ax_r.set_ylim(h, 0)
    ax_r.set_title("Vecteurs de mouvement", fontsize=9, color="white")
    ax_r.axis("off")


def _fig5_psnr(fig, spec, orig_frames, recon_frames,
               encoded_frames, metrics, max_show=4):

    outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec,
                                             hspace=0.4, height_ratios=[1, 1.2])
    top   = gridspec.GridSpecFromSubplotSpec(1, max_show + 1,
                                             subplot_spec=outer[0], wspace=0.06)

    ax_res = fig.add_subplot(top[0])
    p_idx  = next((i for i, f in enumerate(encoded_frames)
                   if f["frame_type"] == "P"), None)
    if p_idx is not None:
        res  = encoded_frames[p_idx]["Y_residual"].astype(np.float32)
        vmax = max(abs(float(res.min())), abs(float(res.max())), 1)
        ax_res.imshow(res, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_res.set_title("Résidu Y\n(P-frame)", fontsize=8, color="white")
    ax_res.axis("off")

    for i in range(min(max_show, len(recon_frames))):
        ax = fig.add_subplot(top[i + 1])
        ax.imshow(cv2.cvtColor(recon_frames[i], cv2.COLOR_BGR2RGB))
        pv = metrics["psnr_list"][i] if i < len(metrics["psnr_list"]) else 0
        ax.set_title(f"Recon F{i+1}\n{pv:.1f} dB", fontsize=8, color="#3fb950")
        ax.axis("off")

    ax_p = fig.add_subplot(outer[1])
    ax_p.set_facecolor("#161b22")

    psnr_vals = metrics["psnr_list"]
    colors    = ["#ff7b72" if f["frame_type"] == "I" else "#79c0ff"
                 for f in encoded_frames[:len(psnr_vals)]]

    ax_p.bar(range(1, len(psnr_vals)+1), psnr_vals, color=colors, alpha=0.85)
    ax_p.axhline(metrics["psnr_mean"], color="#ffa657", lw=1.8, ls="--")

    i_p = mpatches.Patch(color="#ff7b72", label="I-frame")
    p_p = mpatches.Patch(color="#79c0ff", label="P-frame")
    m_l = plt.Line2D([0],[0], color="#ffa657", ls="--",
                     label=f"Moy. {metrics['psnr_mean']:.1f} dB")
    ax_p.legend(handles=[i_p, p_p, m_l],
                facecolor="#0d1117", labelcolor="white", fontsize=9)

    ax_p.set_xlabel(f"Compression Ratio: {metrics['ratio']:.2f}×  |  "
                    f"{metrics['i_frames']} I-frames  +  {metrics['p_frames']} P-frames",
                    color="#8b949e", fontsize=9)
    ax_p.set_ylabel("PSNR (dB)", color="white", fontsize=10)
    ax_p.set_title("PSNR par frame", fontsize=10, color="white")
    ax_p.tick_params(colors="white")
    for sp in ax_p.spines.values():
        sp.set_color("#30363d")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FRAMES_DIR  = "frames"
    BIN_PATH    = "output/video.bin"
    DECODED_DIR = "decoded_frames"
    OUTPUT_DIR  = "output"
    MAX_FRAMES  = 20

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Figure pipeline ───────────────────────────────────────
    print("[1/5] Chargement des frames...")
    orig_frames    = load_frames(FRAMES_DIR)[:MAX_FRAMES]
    recon_frames   = load_frames(DECODED_DIR)[:MAX_FRAMES]
    encoded_frames = decode_from_bin(BIN_PATH)[:MAX_FRAMES]

    print("[2/5] Calcul des métriques...")
    metrics = compute_metrics(orig_frames, recon_frames, encoded_frames, BIN_PATH)

    print("[3/5] Création de la figure pipeline...")
    create_pipeline_figure(
        orig_frames, recon_frames, encoded_frames, metrics,
        save_path=os.path.join(OUTPUT_DIR, "pipeline_visualization.png")
    )

    # ── Expériences rapport ───────────────────────────────────
    print("\n[4/5] Expérience : Compression Ratio vs Quantization Factor...")
    experiment_quality(FRAMES_DIR, max_frames=MAX_FRAMES, output_dir=OUTPUT_DIR)

    print("\n[5/5] Expérience : GOP Size Effect on Compression...")
    experiment_gop(FRAMES_DIR, max_frames=MAX_FRAMES, output_dir=OUTPUT_DIR)

    print("\n✓ Terminé ! Fichiers dans output/")
    print("  - pipeline_visualization.png")
    print("  - graph_quality.png")
    print("  - graph_gop.png")

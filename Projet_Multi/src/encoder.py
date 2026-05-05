"""
encoder.py
----------
Main encoder pipeline: frames folder → compressed .bin file.

Usage
-----
python encoder.py --frames frames/ --output output/video.bin \
                  --gop 10 --search 8 --quality 50
"""

import argparse
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing    import load_frames, bgr_to_ycbcr
from inter_coding     import encode_gop
from entropy_coding   import encode_to_bin


def encode(
    frames_dir: str,
    output_path: str,
    gop_size: int      = 10,
    search_window: int = 8,
    quality: float     = 50.0,
    max_frames: int    = 0,
) -> None:
    # 1. Load frames
    print(f"\n[1/3] Loading frames from '{frames_dir}' …")
    bgr_frames = load_frames(frames_dir)
    if max_frames > 0:
        bgr_frames = bgr_frames[:max_frames]
    print(f"      {len(bgr_frames)} frames loaded.")

    # 2. Pre-process: BGR → YCbCr + 4:2:0
    print("[2/3] Pre-processing (BGR→YCbCr, 4:2:0 subsampling) …")
    ycbcr_frames = [bgr_to_ycbcr(f) for f in bgr_frames]

    # 3. Encode (I-frames + P-frames) + entropy coding
    print(f"[3/3] Encoding | GOP={gop_size} | Search=±{search_window}px | Quality={quality}")
    encoded_frames = encode_gop(
        ycbcr_frames,
        gop_size=gop_size,
        search_window=search_window,
        quality=quality,
    )

    # 4. Write .bin
    compressed_size = encode_to_bin(encoded_frames, output_path)

    # Summary
    n_i = sum(1 for f in encoded_frames if f["frame_type"] == "I")
    n_p = len(encoded_frames) - n_i
    print(f"\nDone! {len(encoded_frames)} frames ({n_i} I, {n_p} P) → {output_path}")
    print(f"Compressed size: {compressed_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPEG-4-like video encoder")
    parser.add_argument("--frames",     default="frames",       help="Input frames directory")
    parser.add_argument("--output",     default="output/video.bin", help="Output .bin file")
    parser.add_argument("--gop",        type=int,   default=10, help="GOP size (I-frame interval)")
    parser.add_argument("--search",     type=int,   default=8,  help="Motion search window ±S")
    parser.add_argument("--quality",    type=float, default=50, help="Quantisation quality (1–100)")
    parser.add_argument("--max_frames", type=int,   default=0,  help="Limit number of frames (0=all)")
    args = parser.parse_args()

    encode(
        frames_dir=args.frames,
        output_path=args.output,
        gop_size=args.gop,
        search_window=args.search,
        quality=args.quality,
        max_frames=args.max_frames,
    )
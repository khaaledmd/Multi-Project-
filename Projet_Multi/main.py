"""
main.py
-------
Full pipeline entry point:
  1. Extract frames from a video file
  2. Encode frames → .bin
  3. Decode .bin → reconstructed frames
  4. Print basic stats (PSNR, compression ratio)

Usage
-----
python main.py --video my_video.mp4 \
               [--fps 5] [--max_frames 60] \
               [--gop 10] [--search 8] [--quality 50]
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from extract_frames import extract_frames
from encoder        import encode
from decoder        import decode
from preprocessing  import load_frames


def compute_psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description="Simplified MPEG-4 Encoder — full pipeline")
    parser.add_argument("--video",      required=True,               help="Input video file")
    parser.add_argument("--frames_dir", default="frames",            help="Directory for extracted frames")
    parser.add_argument("--bin_file",   default="output/video.bin",  help="Compressed output file")
    parser.add_argument("--decoded_dir",default="decoded_frames",    help="Directory for decoded frames")
    parser.add_argument("--fps",        type=int,   default=5,       help="FPS to extract from video")
    parser.add_argument("--max_frames", type=int,   default=30,      help="Max frames to process")
    parser.add_argument("--gop",        type=int,   default=10,      help="GOP size")
    parser.add_argument("--search",     type=int,   default=8,       help="Motion search window ±S")
    parser.add_argument("--quality",    type=float, default=50.0,    help="Quality factor (1–100)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1 — Extract frames
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("STEP 1 — Frame Extraction")
    print("=" * 60)
    n = extract_frames(
        video_path=args.video,
        output_dir=args.frames_dir,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Encode
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 2 — Encoding")
    print("=" * 60)
    encode(
        frames_dir=args.frames_dir,
        output_path=args.bin_file,
        gop_size=args.gop,
        search_window=args.search,
        quality=args.quality,
        max_frames=args.max_frames,
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Decode
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 3 — Decoding")
    print("=" * 60)
    decode(args.bin_file, args.decoded_dir)

    # ------------------------------------------------------------------ #
    # Step 4 — Basic evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation")
    print("=" * 60)

    orig_frames = load_frames(args.frames_dir)[:args.max_frames]
    dec_frames  = load_frames(args.decoded_dir)

    psnr_values = [
        compute_psnr(o, d)
        for o, d in zip(orig_frames, dec_frames)
    ]
    avg_psnr = np.mean(psnr_values)

    # Compression ratio: total original pixel data vs compressed file
    sample = orig_frames[0]
    frame_bytes = sample.shape[0] * sample.shape[1] * sample.shape[2]  # H*W*3
    total_original = frame_bytes * len(orig_frames)
    compressed_size = os.path.getsize(args.bin_file)
    ratio = total_original / compressed_size

    print(f"Frames          : {len(orig_frames)}")
    print(f"Average PSNR    : {avg_psnr:.2f} dB")
    print(f"Original size   : {total_original:,} bytes")
    print(f"Compressed size : {compressed_size:,} bytes")
    print(f"Compression ratio: {ratio:.2f}×")


if __name__ == "__main__":
    main()
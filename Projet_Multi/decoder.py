"""
decoder.py
----------
Main decoder pipeline: .bin file → reconstructed frames.

Usage
-----
python decoder.py --input output/video.bin --output decoded_frames/
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from inter_coding   import decode_gop
from entropy_coding import decode_from_bin
from preprocessing  import ycbcr_to_bgr


def decode(bin_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/2] Reading compressed file: {bin_path}")
    encoded_frames = decode_from_bin(bin_path)

    print("[2/2] Decoding frames …")
    decoded_ycbcr = decode_gop(encoded_frames)

    for idx, (Y, Cb, Cr) in enumerate(decoded_ycbcr):
        bgr = ycbcr_to_bgr(Y, Cb, Cr)
        out_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, bgr)

    print(f"\nDone! {len(decoded_ycbcr)} frames saved → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPEG-4-like video decoder")
    parser.add_argument("--input",  default="output/video.bin", help="Input .bin file")
    parser.add_argument("--output", default="decoded_frames",   help="Output directory for frames")
    args = parser.parse_args()

    decode(args.input, args.output)
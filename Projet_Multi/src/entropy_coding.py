"""
entropy_coding.py
-----------------
Part 4 — Entropy Coding
  • Serialise all frame dicts into bytes using pickle
  • Apply zlib lossless compression (deflate)
  • Write to / read from a .bin file
"""

import pickle
import zlib
import os
from typing import List


def encode_to_bin(encoded_frames: List[dict], output_path: str) -> int:
    """
    Serialise and compress encoded frame dicts to a .bin file.

    Returns the size (bytes) of the compressed file.
    """
    raw_bytes = pickle.dumps(encoded_frames)
    compressed = zlib.compress(raw_bytes, level=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(compressed)

    print(f"Written → {output_path}")
    print(f"  Raw bytes    : {len(raw_bytes):,}")
    print(f"  Compressed   : {len(compressed):,}")
    print(f"  Ratio        : {len(raw_bytes)/len(compressed):.2f}×")
    return len(compressed)


def decode_from_bin(bin_path: str) -> List[dict]:
    """
    Read and decompress a .bin file back into a list of frame dicts.
    """
    with open(bin_path, "rb") as f:
        compressed = f.read()

    raw_bytes = zlib.decompress(compressed)
    encoded_frames = pickle.loads(raw_bytes)
    print(f"Loaded {len(encoded_frames)} frames from {bin_path}")
    return encoded_frames
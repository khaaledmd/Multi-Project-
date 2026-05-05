"""
extract_frames.py
-----------------
Extracts frames from a video file and saves them as PNG images.
Usage: python src/extract_frames.py --video <path_to_video> [--fps 5] [--max_frames 60]
"""

import cv2
import os
import argparse


def extract_frames(video_path: str, output_dir: str, fps: int = 5, max_frames: int = 60) -> int:
    """
    Extract frames from a video file.

    Args:
        video_path:  Path to the input video file.
        output_dir:  Directory where extracted frames will be saved.
        fps:         Target frames per second to extract.
        max_frames:  Maximum number of frames to extract.

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # fallback

    frame_interval = max(1, round(video_fps / fps))
    print(f"Video FPS: {video_fps:.2f} | Extracting every {frame_interval} frame(s) → ~{fps} fps")

    frame_idx = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames → {output_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video",      required=True,  help="Path to the input video file")
    parser.add_argument("--output_dir", default="frames", help="Output directory for frames")
    parser.add_argument("--fps",        type=int, default=5,  help="Target FPS to extract")
    parser.add_argument("--max_frames", type=int, default=60, help="Max number of frames")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.fps, args.max_frames)
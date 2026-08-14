"""
Convert a StreamingRecorderManager HDF5 dataset into viewable .mp4 videos,
one per recorded episode.

Schema, confirmed directly from hdf5_dataset_file_handler.py:
    /data                          (group, attrs: total=<int total steps recorded>)
      /data/demo_0                (group, one per episode written to disk)
          attrs: num_samples=<int>, seed=<int, optional>, success=<bool, optional>
          side_camera_rgb          dataset, shape [T, H, W, 3] (per-env; confirm
                                   on your first real file — see note below)
      /data/demo_1
          ...

Pure post-processing — no Isaac Sim / AppLauncher required. Run this after
training, pointed at whatever dataset_export_dir_path/dataset_filename your
RecordCfg wrote to.
"""

import argparse
import os

import h5py
import cv2
import numpy as np


def export_demo_video(h5_episode_group: h5py.Group, demo_name: str, out_dir: str, fps: int) -> bool:
    if "side_camera_rgb" not in h5_episode_group:
        print(f"  [{demo_name}] no 'side_camera_rgb' key found — skipping")
        return False

    frames = h5_episode_group["side_camera_rgb"][:]

    # frames may come back as [T, H, W, 3] (already per-env) or [T, N, H, W, 3]
    # (still carrying an env dimension) depending on how the term stored it —
    # handle both rather than assuming.
    if frames.ndim == 5:
        frames = frames[:, 0]  # take env index 0
    elif frames.ndim != 4:
        print(f"  [{demo_name}] unexpected frame shape {frames.shape} — skipping")
        return False

    if frames.shape[0] == 0:
        print(f"  [{demo_name}] empty frame array — skipping")
        return False

    success = h5_episode_group.attrs.get("success", None)
    tag = "success" if success else ("fail" if success is not None else "unknown")

    out_path = os.path.join(out_dir, f"{demo_name}_{tag}.mp4")
    h, w = frames.shape[1], frames.shape[2]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

    print(f"  [{demo_name}] success={success}, num_samples={h5_episode_group.attrs.get('num_samples')}, "
          f"frames={frames.shape[0]} -> wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser("Extract videos from a RecorderManager HDF5 dataset")
    parser.add_argument("--hdf5_path", type=str, help="path to the .hdf5 file", default="logs/recordings2/dataset.hdf5")
    parser.add_argument("--out_dir", type=str, default="extracted_videos")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--filter", type=str, default="all",
                         choices=["all", "success", "fail"],
                         help="only export episodes matching this outcome")
    parser.add_argument("--limit", type=int, default=None,
                         help="only export the first N matching episodes (useful for a quick spot-check)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.hdf5_path, "r") as f:
        if "data" not in f:
            print("no 'data' group found — is this really a RecorderManager HDF5 file?")
            return

        data_group = f["data"]
        print(f"total steps recorded (attrs['total']): {data_group.attrs.get('total')}")

        demo_names = sorted(
            [k for k in data_group.keys() if k.startswith("demo_")],
            key=lambda name: int(name.split("_")[1]),
        )
        print(f"found {len(demo_names)} episodes total")

        exported = 0
        for demo_name in demo_names:
            episode_group = data_group[demo_name]
            success = episode_group.attrs.get("success", None)

            if args.filter == "success" and not success:
                continue
            if args.filter == "fail" and (success or success is None):
                continue

            if export_demo_video(episode_group, demo_name, args.out_dir, args.fps):
                exported += 1

            if args.limit is not None and exported >= args.limit:
                break

        print(f"\nexported {exported} episode(s) matching filter='{args.filter}' to {args.out_dir}/")


if __name__ == "__main__":
    main()
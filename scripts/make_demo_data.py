#!/usr/bin/env python3
"""Create a synthetic demo project (psifx-format pose archives + WAKEE annotations).

    python scripts/make_demo_data.py demo
    vas -c demo/configs/config.yaml run

No GPU, ffmpeg or psifx is required: video processing is skipped.
"""

import argparse

from vas.synthetic import make_demo_project

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="demo")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--duration", type=float, default=90.0, help="seconds per session")
    args = ap.parse_args()
    cfg = make_demo_project(args.root, n_sessions=args.sessions, duration_s=args.duration,
                            config_overrides={"flags": {"skip_pose_clustering": False}})
    print(f"Demo project created. Run:\n  vas -c {cfg} run")

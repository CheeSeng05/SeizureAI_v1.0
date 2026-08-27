from __future__ import annotations

import argparse
import subprocess
import sys


def run(module: str, config: str) -> None:
    print(f"\n=== Running {module} ===")
    subprocess.run([sys.executable, "-m", module, "--config", config], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run annotation adaptation, video pose processing, windowing and training."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--through",
        choices=["annotations", "pose", "features", "dataset", "train", "evaluate"],
        default="evaluate",
    )
    args = parser.parse_args()

    stages = [
        ("annotations", "body_pose.prepare_annotations"),
        ("pose", "body_pose.extract_pose"),
        ("features", "body_pose.extract_features"),
        ("dataset", "body_pose.build_dataset"),
        ("train", "body_pose.train_lstm"),
        ("evaluate", "body_pose.evaluate"),
    ]
    for stage_name, module in stages:
        run(module, args.config)
        if stage_name == args.through:
            break


if __name__ == "__main__":
    main()

"""
CLI inference entry point.
"""

from __future__ import annotations

import argparse
import json

from src.inference import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run personal color and style recommendation for one portrait.")
    parser.add_argument("image_path", type=str, help="Path to portrait image.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference(args.image_path)
    printable = {
        "cluster_id": result["cluster_id"],
        "cluster_label": result["cluster_label"],
        "features": result["features"],
        "explanation": result["explanation"],
        "recommendations": result["recommendations"],
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Command line: python -m id_classifier <command> ...

    build-ground-truth   write ground_truth.csv from the labeled NFS folders (distinct pictures only)
    evaluate             detect + classify + route every ground-truth image and write the results report

See README.md for the full walk-through.
"""

from __future__ import annotations

import argparse
import logging
import sys

from id_classifier.evaluate import run_experiment
from id_classifier.ground_truth import build_ground_truth


def cmd_build_ground_truth(args: argparse.Namespace) -> int:
    csv_path = build_ground_truth(args.root, args.out, per_folder=args.per_folder, seed=args.seed)
    print(f"Wrote {csv_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = run_experiment(args.config)
    print(f"Results: {result.output_dir / 'comparison.md'}")
    print(f"Winner: {result.winner or 'none (no configuration meets the selection rule)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m id_classifier", description="Classify non-Egyptian ID images.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build-ground-truth", help="write ground_truth.csv from the labeled NFS folders")
    p.add_argument("--root", default="/mnt/nfs", help="NFS root as this machine sees it (default: /mnt/nfs)")
    p.add_argument("--out", default="data/ground_truth.csv", help="output CSV (default: data/ground_truth.csv)")
    p.add_argument("--per-folder", type=int, help="at most this many DISTINCT pictures per folder (default: all)")
    p.add_argument("--seed", type=int, default=42, help="same seed = same sample (default: 42)")
    p.set_defaults(func=cmd_build_ground_truth)

    p = commands.add_parser("evaluate", help="run the configuration(s) of an experiment YAML and write the report")
    p.add_argument("--config", required=True, help="experiment YAML, e.g. configs/nfs_eval.yaml")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

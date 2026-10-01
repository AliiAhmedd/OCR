"""Command line: python -m id_classifier <command> ...

    generate-synthetic   write fake ID images + ground_truth.csv (no real personal data)
    evaluate             run every configuration of an experiment YAML and write the comparison table
    classify             run the pipeline on a source and store the results in the database
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from id_classifier.config import build_classifiers, build_detector, build_source, load_yaml
from id_classifier.evaluate import run_experiment
from id_classifier.pipeline import new_run_id, run_classification
from id_classifier.routing import RoutingConfig
from id_classifier.storage import Storage
from id_classifier.synthetic import SyntheticSource, write_dataset


def cmd_generate_synthetic(args: argparse.Namespace) -> int:
    source = SyntheticSource(per_country=args.per_country, negatives=args.negatives, seed=args.seed)
    csv_path = write_dataset(source, args.out)
    per_type = Counter(s.document_type for s in source.samples)
    print(f"Wrote {len(source.samples)} images and {csv_path}")
    print(f"Per document type: {dict(per_type)}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = run_experiment(args.config)
    print(f"Results: {result.output_dir / 'comparison.md'}")
    print(f"Winner: {result.winner or 'none (no configuration meets the selection rule)'}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    if "classify" not in config:
        raise SystemExit(f"{args.config} has no 'classify' section")
    section = config["classify"]
    counts = run_classification(
        source=build_source(section["source"]),
        detector=build_detector(section.get("detector")),
        classifiers=build_classifiers(section["classifiers"]),
        routing=RoutingConfig.from_dict(config.get("routing")),
        storage=Storage(section["database_url"]),
        run_id=args.run_id or new_run_id(),
    )
    print(f"Run {counts['run_id']}: {counts['images']} images")
    print(f"Per country: {dict(sorted(counts['per_country'].items()))}")
    print(f"Per routing status: {dict(sorted(counts['per_status'].items()))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m id_classifier", description="Classify non-Egyptian ID images.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("generate-synthetic", help="write fake ID images and a ground-truth CSV")
    p.add_argument("--out", default="data/synthetic", help="output folder (default: data/synthetic)")
    p.add_argument("--per-country", type=int, default=4, help="ID cards per country, front+back (default: 4)")
    p.add_argument("--negatives", type=int, default=8, help="images without a document (default: 8)")
    p.add_argument("--seed", type=int, default=42, help="same seed = same images (default: 42)")
    p.set_defaults(func=cmd_generate_synthetic)

    p = commands.add_parser("evaluate", help="compare the configurations of an experiment YAML")
    p.add_argument("--config", required=True, help="experiment YAML, e.g. configs/demo.yaml")
    p.set_defaults(func=cmd_evaluate)

    p = commands.add_parser("classify", help="run the pipeline and store results in the database")
    p.add_argument("--config", required=True, help="YAML with a 'classify' section, e.g. configs/demo.yaml")
    p.add_argument("--run-id", help="reuse a run id to rerun safely (default: cli-<timestamp>)")
    p.set_defaults(func=cmd_classify)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

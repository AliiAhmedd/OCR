"""Tune the kNN decision + routing settings without re-running the pipeline.

A full sweep re-runs detector + embedding model for every configuration (~2 min each on CPU), although the kNN
settings (floor, 1-NN similarity / margin / trigger share) and the routing threshold only change the LAST step.
So:

1. Embedding pass, once: every ground-truth image is fetched from the experiment's `source:` (the same storage
   and the same references as `evaluate`), the detector runs, every region is embedded. Cached in data/cache,
   so a second grid on the same ground truth + source skips even this.
2. Every setting of the grid: the real EmbeddingKnnClassifier.classify_vector -> route_image -> summarize_image
   on the stored vectors (milliseconds). One row per setting is appended to results.md / results.csv as soon as
   it is done, so the tuning can be followed live in a file.

Goal columns (progress.md rows 26-27): a document whose class (type/country/side) has 2+ images still has a
reference left under leave-one-out, so it could be recognised; ideally only classes with ONE image go to review.
That must not be bought with documents auto-accepted under the wrong country (accepted_wrong_country = 0).

    python -m id_classifier tune-routing --grid configs/routing_grid.yaml
    python -m id_classifier tune-routing --grid configs/routing_grid.yaml --check-run data/runs/<sweep run>
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from id_classifier.config import build_classifiers, build_detector, build_source, load_yaml
from id_classifier.evaluate import _fmt as _csv_fmt, _slug, class_sizes, read_ground_truth, review_single_sample
from id_classifier.classifiers.knn import ROTATIONS, rotations_for
from id_classifier.routing import RoutingConfig, route_image, summarize_image
from id_classifier.sources import read_image_file
from id_classifier.types import NO_DOCUMENT, Region

log = logging.getLogger(__name__)

KNN_SETTINGS = ("k", "min_similarity", "nn_fallback", "nn_min_similarity", "nn_margin", "nn_trigger_share",
                "nn_confidence", "query_rotations", "nn_max_class_refs",
                "nn_big_class_min_similarity", "nn_side_check")
COLUMNS = ["setting", "coverage", "accepted_accuracy", "majority_accepted_accuracy", "accepted_wrong_country",
           "reviewed_multi_sample",
           "reviewed_single_sample", "beyond_accepted", "beyond_correct", "negatives_rejected", "goal"]


@dataclass
class StoredImage:
    """One ground-truth image after the embedding pass: what routing needs, no pixels."""
    record: object                       # ImageRecord with image=None (fetch_status / failure kept)
    regions: list[Region]                # crops dropped
    vectors: list[np.ndarray]            # per region: one embedding per turn in ROTATIONS (shape 4 x d)


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def embedding_pass(truths, detector, knn, source, cache: Path | None) -> list[StoredImage]:
    """Fetch -> detect -> embed every region, once. Cached as <cache>.npz (vectors) + <cache>.json (regions)."""
    from id_classifier.types import ImageRecord

    if cache is not None and cache.with_suffix(".json").exists():
        meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        vectors = np.load(cache.with_suffix(".npz"))["vectors"]
        images, i = [], 0
        for m in meta:
            record = ImageRecord(m["transaction_id"], m["image_id"], m["reference"], fetch_status=m["fetch_status"],
                                 failure_reason=m.get("failure_reason"))
            regions = [Region(**r) for r in m["regions"]]
            images.append(StoredImage(record, regions, [vectors[i + j] for j in range(len(regions))]))   # 4 x d each
            i += len(regions)
        log.info("Embedding pass: %d images loaded from %s", len(images), cache.with_suffix(".npz"))
        return images

    images = []
    for number, t in enumerate(truths, start=1):
        record = (read_image_file(t.transaction_id, t.image_id, t.image_path) if source is None
                  else source.fetch(t.transaction_id, t.image_id, t.reference))
        regions = detector.detect(record) if record.fetch_status == "ok" else []
        vectors = [np.stack([knn.embed(r.crop.rotate(a, expand=True) if a else r.crop) for a in ROTATIONS])
                   for r in regions]                       # every turn, so any query_rotations can be tuned
        record.image = None
        for r in regions:
            r.crop = None
        images.append(StoredImage(record, regions, vectors))
        if number % 25 == 0 or number == len(truths):
            log.info("Embedding pass: %d/%d images", number, len(truths))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        all_vectors = [v for img in images for v in img.vectors]
        np.savez(cache.with_suffix(".npz"), vectors=np.stack(all_vectors) if all_vectors else np.zeros((0, 4, 1)))
        meta = [{"transaction_id": img.record.transaction_id, "image_id": img.record.image_id,
                 "reference": img.record.image_reference, "fetch_status": img.record.fetch_status,
                 "failure_reason": img.record.failure_reason,
                 "regions": [{"region_id": r.region_id, "transaction_id": r.transaction_id, "image_id": r.image_id,
                              "bbox": list(r.bbox), "confidence": r.confidence, "detector_name": r.detector_name}
                             for r in img.regions]} for img in images]
        cache.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return images


def _classify_stored(knn, region: Region, views: np.ndarray):
    """classify() on stored turns: the same turns rotations_for() would embed for this crop."""
    x0, y0, x1, y1 = region.bbox
    angles = rotations_for(x1 - x0, y1 - y0, knn.query_rotations)
    chosen = views[[ROTATIONS.index(a) for a in angles]]
    return knn.classify_vector(chosen if len(angles) > 1 else chosen[0], angles=angles)


def route_all(images: list[StoredImage], knn, routing: RoutingConfig, detector_name: str, truths=None,
              sizes=None) -> list:
    """The image summaries for the knn's CURRENT settings: real classify_vector + route_image + summarize_image,
    then the single-sample evaluation rule when `sizes` is given (as evaluate does)."""
    summaries = []
    for i, img in enumerate(images):
        predictions = {r.region_id: [_classify_stored(knn, r, v)] for r, v in zip(img.regions, img.vectors)}
        summary = summarize_image(route_image(img.record, img.regions, predictions, routing, detector_name))
        if sizes is not None:
            review_single_sample(truths[i], summary, sizes)
        summaries.append(summary)
    return summaries


def rule_sizes(experiment: dict, truths):
    """Class sizes when the experiment switches the single-sample evaluation rule on, else None."""
    return class_sizes(truths) if experiment.get("review_single_sample_classes") else None


def score(truths, summaries) -> dict:
    docs = [(t, s) for t, s in zip(truths, summaries) if t.is_document]
    sizes = Counter((t.document_type, t.issuing_country, t.document_side) for t, _ in docs)
    majority = Counter(t.issuing_country for t, _ in docs).most_common(1)[0][0]
    right = lambda t, s: (s.document_type, s.issuing_country, s.document_side) == \
                         (t.document_type, t.issuing_country, t.document_side)
    accepted = [(t, s) for t, s in docs if not s.needs_review]
    reviewed = [(t, s) for t, s in docs if s.needs_review]
    beyond = [(t, s) for t, s in accepted if t.issuing_country != majority]
    in_majority = [(t, s) for t, s in accepted if t.issuing_country == majority]
    multi = [(t, s) for t, s in reviewed if sizes[(t.document_type, t.issuing_country, t.document_side)] >= 2]
    wrong_country = sum(s.issuing_country != t.issuing_country for t, s in accepted)
    negatives = [(t, s) for t, s in zip(truths, summaries) if not t.is_document]
    return {
        "coverage": len(accepted) / len(docs) if docs else None,
        "accepted_accuracy": sum(right(t, s) for t, s in accepted) / len(accepted) if accepted else None,
        "majority_accepted_accuracy": (sum(right(t, s) for t, s in in_majority) / len(in_majority)
                                       if in_majority else None),
        "accepted_wrong_country": wrong_country,
        "reviewed_multi_sample": len(multi),
        "reviewed_single_sample": len(reviewed) - len(multi),
        "beyond_accepted": len(beyond),
        "beyond_correct": sum(right(t, s) for t, s in beyond),
        "negatives_rejected": f"{sum(s.document_type == NO_DOCUMENT for t, s in negatives)}/{len(negatives)}",
        "goal": "YES" if not multi and not wrong_country else "",
        "_majority": majority,
        "_multi": [f"{t.document_type}/{t.issuing_country}/{t.document_side} (n={sizes[(t.document_type, t.issuing_country, t.document_side)]}) "
                   f"-> {s.document_type}/{s.issuing_country}/{s.document_side} [{','.join(s.reasons)}]" for t, s in multi],
    }


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else ("n/a" if v is None else str(v))


def _setting_name(values: dict) -> str:
    short = {"min_similarity": "floor", "nn_min_similarity": "s", "nn_margin": "m", "nn_trigger_share": "share",
             "confidence_threshold": "thr", "nn_fallback": "nn", "nn_confidence": "nnconf", "k": "k",
             "query_rotations": "rot", "nn_max_class_refs": "maxrefs",
             "nn_big_class_min_similarity": "bigs", "nn_side_check": "side"}
    return " ".join(f"{short.get(k, k)}={v}" for k, v in values.items())


def grid_settings(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def apply(knn, base: dict, values: dict, routing_base: dict) -> RoutingConfig:
    for name in KNN_SETTINGS:
        setattr(knn, name, values.get(name, base.get(name, getattr(knn, name))))
    return RoutingConfig.from_dict({**routing_base, **{k: v for k, v in values.items() if k == "confidence_threshold"}})


def load(experiment_path: str | Path):
    """Ground truth, source, detector and kNN (references from the source, cached) of an experiment YAML."""
    experiment = load_yaml(experiment_path)
    truths = read_ground_truth(experiment["ground_truth"])
    source = build_source(experiment.get("source"))
    spec = experiment["configurations"][0]
    detector = build_detector(spec["detector"])
    knn_spec = spec["classifiers"][0]
    [knn] = build_classifiers([knn_spec], source)
    cache = Path(knn_spec.get("cache_dir", "data/cache")) / "routing_queries_{}".format(_key(
        Path(experiment["ground_truth"]).read_text(encoding="utf-8"), spec["detector"], knn_spec.get("model"),
        getattr(source, "name", None), ROTATIONS))
    return experiment, truths, source, detector, knn, knn_spec, cache


KNN_DEFAULTS = {"nn_fallback": False, "min_similarity": 0.5, "nn_min_similarity": 0.8, "nn_margin": 0.05,
                "nn_trigger_share": 0.8, "nn_confidence": 0.9, "k": 5, "query_rotations": "none",
                "nn_max_class_refs": None, "nn_big_class_min_similarity": None,
                "nn_side_check": False}   # EmbeddingKnnClassifier defaults


def configure(knn, knn_spec: dict, spec: dict, experiment: dict) -> RoutingConfig:
    """Puts the kNN in the settings of one configuration of the experiment (spec) and returns its routing."""
    settings = {k: v for k, v in spec["classifiers"][0].items() if k in KNN_SETTINGS}
    base = {**KNN_DEFAULTS, **{k: v for k, v in knn_spec.items() if k in KNN_SETTINGS}}
    return apply(knn, base, settings, experiment.get("routing") or {})


def check_run(images, truths, knn, knn_spec, experiment, detector_name: str, run_dir: Path) -> list[str]:
    """Replays every configuration of the experiment and compares with the run's predictions_<config>.csv,
    image for image. Returns one line per configuration."""
    lines = []
    for spec in experiment["configurations"]:
        routing = configure(knn, knn_spec, spec, experiment)
        run_experiment_yaml = load_yaml(run_dir / "experiment.yaml") if (run_dir / "experiment.yaml").exists() else experiment
        summaries = route_all(images, knn, routing, detector_name, truths, rule_sizes(run_experiment_yaml, truths))
        with (run_dir / f"predictions_{_slug(spec['name'])}.csv").open(newline="", encoding="utf-8") as f:
            saved = list(csv.DictReader(f))
        diff = [t.image_id + str(t.transaction_id) for t, s, row in zip(truths, summaries, saved)
                if tuple(map(_csv_fmt, (s.document_type, s.issuing_country, s.document_side, s.needs_review)))
                != (row["pred_type"], row["pred_country"], row["pred_side"], row["needs_review"])]
        lines.append(f"{spec['name']}: {len(saved) - len(diff)}/{len(saved)} images identical"
                     + (f", differ: {diff[:5]}" if diff else ""))
    return lines


def run(grid_path: str | Path, out_dir: str | Path | None = None, check: str | Path | None = None) -> Path:
    grid_file = load_yaml(grid_path)
    experiment, truths, source, detector, knn, knn_spec, cache = load(grid_file["experiment"])
    images = embedding_pass(truths, detector, knn, source, cache)

    out = Path(out_dir or f"data/sweeps/routing_tune_{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    md, csv_path = out / "results.md", out / "results.csv"

    if check:
        lines = check_run(images, truths, knn, knn_spec, experiment, detector.name, Path(check))
        (out / "check.md").write_text("# Replay check\n\nRun: `{}`\n\n{}\n".format(
            check, "\n".join(f"- {l}" for l in lines)), encoding="utf-8")
        for l in lines:
            log.info("check: %s", l)

    base = {**{k: v for k, v in knn_spec.items() if k in KNN_SETTINGS}, **(grid_file.get("fixed") or {})}
    sizes = rule_sizes(experiment, truths)
    settings = grid_settings(grid_file["grid"])
    md.write_text("\n".join([
        f"# Routing tune: {Path(grid_path).name}",
        "",
        f"- Experiment: `{grid_file['experiment']}` (ground truth `{experiment['ground_truth']}`, {len(truths)} images, "
        f"read from {getattr(source, 'name', 'local files')})",
        f"- Fixed: {base}; routing {experiment.get('routing') or {}}",
        f"- {len(settings)} settings. One row per setting, appended as soon as it is computed.",
        (f"- Evaluation rule review_single_sample_classes: ON ({sum(n for n in sizes.values() if n < 2)} documents of "
         "one-image classes always go to review)" if sizes else "- Evaluation rule review_single_sample_classes: off"),
        "- coverage / accepted_accuracy: over all documents. accepted_wrong_country: auto-accepted documents with "
        "the WRONG country (count). reviewed_multi_sample: reviewed documents whose class (type/country/side) has 2+ "
        "images (could be recognised); reviewed_single_sample: class with 1 image (can never be, leave-one-out). "
        "beyond_*: auto-accepted / correct outside the biggest country. goal: reviewed_multi_sample = 0 and "
        "accepted_wrong_country = 0.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
        "",
    ]), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(COLUMNS)

    results = []
    for number, values in enumerate(settings, start=1):
        routing = apply(knn, base, values, experiment.get("routing") or {})
        summaries = route_all(images, knn, routing, detector.name, truths, sizes)
        result = {"setting": _setting_name(values), **score(truths, summaries)}
        results.append(result)
        with md.open("a", encoding="utf-8") as f:
            f.write("| " + " | ".join(_fmt(result[c]) for c in COLUMNS) + " |\n")
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([result[c] for c in COLUMNS])
        if number % 20 == 0 or number == len(settings):
            log.info("Routing tune: %d/%d settings done -> %s", number, len(settings), md)

    # Best: no wrong-country accept first, then fewest multi-sample documents in review, then most coverage.
    ranked = sorted(results, key=lambda r: (r["accepted_wrong_country"], r["reviewed_multi_sample"],
                                            -(r["coverage"] or 0), -(r["accepted_accuracy"] or 0)))
    reached = [r for r in results if r["goal"]]
    with md.open("a", encoding="utf-8") as f:
        f.write("\n## Best settings\n\nRanked by accepted_wrong_country, then reviewed_multi_sample, then coverage, "
                "then accepted_accuracy.\n\n")
        f.write(f"**Goal reached by {len(reached)} of {len(results)} settings.**\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n")
        for r in ranked[:10]:
            f.write("| " + " | ".join(_fmt(r[c]) for c in COLUMNS) + " |\n")
        f.write(f"\nMulti-sample documents still in review with the best setting ({ranked[0]['setting']}):\n\n")
        f.write("\n".join(f"- {line}" for line in ranked[0]["_multi"]) or "- none")
        f.write("\n")
    floor = (grid_file.get("constraint") or {}).get("min_majority_accepted_accuracy")
    if floor is not None:
        ok = [r for r in results if r["accepted_wrong_country"] == 0
              and (r["majority_accepted_accuracy"] or 0) >= floor]
        ok.sort(key=lambda r: (-r["beyond_correct"], r["reviewed_multi_sample"], -(r["coverage"] or 0)))
        lines = [
            "",
            f"## Constrained best: {results[0]['_majority']} accepted_accuracy >= {floor}, 0 wrong-country",
            "",
            "Ranked by correct auto-accepts outside the biggest country (beyond_correct; single-sample classes "
            "always go to review, so these are multi-sample documents), then fewest multi-sample documents in "
            f"review, then coverage. **{len(ok)} of {len(results)} settings qualify.**",
            "",
            "| " + " | ".join(COLUMNS) + " |",
            "|" + "---|" * len(COLUMNS),
            *("| " + " | ".join(_fmt(r[c]) for c in COLUMNS) + " |" for r in ok[:15]),
        ]
        if ok:
            lines += ["", f"Multi-sample documents still in review with {ok[0]['setting']}:", ""]
            lines += [f"- {line}" for line in ok[0]["_multi"]] or ["- none"]
        with md.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log.info("Constrained best: %d of %d settings qualify -> %s", len(ok), len(results), md)
    log.info("Routing tune done: %s (goal reached by %d of %d settings)", md, len(reached), len(results))
    return md

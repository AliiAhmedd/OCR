"""Look at the crops the kNN classifies, for the documents outside the biggest country (progress.md row 27).

For every such document: the photo with the detector box, the crop, and the crop of its nearest OTHER reference
(leave-one-out, as the kNN sees it), next to the numbers: box size and aspect ratio, detector confidence, whole-image
fallback (no card box found), nearest reference label + similarity, best reference of the true class.

Dev tool only, not part of the package. Uses the regions + embeddings of `tune-routing`'s embedding pass (run it
first) and fetches the images from the experiment's `source:` again (only those shown). The output holds real ID
images: it is written under data/ (git-ignored) and stays on this machine.

    python scripts/inspect_crops.py --experiment configs/azurite_sweep.yaml
    # -> data/crop_review/<time>/index.html (pictures) + crops.csv (numbers only)
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from id_classifier.detectors import FULL_IMAGE_INDEX
from id_classifier.classifiers.knn import ROTATIONS, rotations_for
from id_classifier.routing_tune import _classify_stored, configure, embedding_pass, load, route_all, rule_sizes


def _data_uri(image, max_side: int = 360) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _fetch(source, truth):
    from id_classifier.sources import read_image_file

    return (read_image_file(truth.transaction_id, truth.image_id, truth.image_path) if source is None
            else source.fetch(truth.transaction_id, truth.image_id, truth.reference))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="configs/azurite_sweep.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="every document, not only outside the biggest country")
    parser.add_argument("--configuration", default="nn_s0.85_m0.02",
                        help="configuration of the experiment whose routing is shown (default: the Azurite sweep winner)")
    parser.add_argument("--only-1nn-failures", action="store_true",
                        help="every image (any country) where the 1-NN fallback came into play and the result failed "
                             "(review, or auto-accepted with a wrong label)")
    parser.add_argument("--query-rotations", choices=["none", "portrait", "portrait_box", "all"],
                        help="override the configuration's query_rotations (knn.rotations_for)")
    args = parser.parse_args(argv)

    experiment, truths, source, detector, knn, knn_spec, cache = load(args.experiment)
    images = embedding_pass(truths, detector, knn, source, cache)
    if len(knn.labels) != len(truths):
        print(f"kNN has {len(knn.labels)} references for {len(truths)} ground-truth rows: cannot pair them")
        return 1

    [spec] = [c for c in experiment["configurations"] if c["name"] == args.configuration]
    routing = configure(knn, knn_spec, spec, experiment)
    if args.query_rotations:
        knn.query_rotations = args.query_rotations
    sizes_rule = rule_sizes(experiment, truths)
    summaries = route_all(images, knn, routing, detector.name, truths, sizes_rule)

    label = lambda t: f"{t.document_type}/{t.issuing_country}/{t.document_side}"
    sizes = Counter(label(t) for t in truths)
    majority = Counter(t.issuing_country for t in truths if t.is_document).most_common(1)[0][0]
    raws = [json.loads(_classify_stored(knn, img.regions[0], img.vectors[0]).raw_output) for img in images]
    def right(i: int) -> bool:
        s, t = summaries[i], truths[i]
        return (s.document_type, s.issuing_country, s.document_side) == (t.document_type, t.issuing_country, t.document_side)
    failed = lambda i: summaries[i].needs_review or not right(i)
    if args.only_1nn_failures:            # the 1-NN fallback was tried (taken, rejected, side mismatch, big class)
        chosen = [i for i in range(len(truths)) if "1nn" in raws[i].get("decision", "") and failed(i)]
        chosen.sort(key=lambda i: (summaries[i].needs_review, sizes[label(truths[i])] < 2))   # wrong accepts first
    else:
        chosen = [i for i, t in enumerate(truths) if args.all or (t.is_document and t.issuing_country != majority)]
        chosen.sort(key=lambda i: sizes[label(truths[i])] < 2)      # classes with 2+ images first: those matter

    out = args.out or Path(f"data/crop_review/{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    for i in chosen:
        t, img = truths[i], images[i]
        region, views = img.regions[0], img.vectors[0]                 # 4 turns of the crop (ROTATIONS)
        bx0, by0, bx1, by1 = region.bbox
        angles = rotations_for(bx1 - bx0, by1 - by0, knn.query_rotations, region.region_id.endswith(f":{FULL_IMAGE_INDEX}"))
        turned = views[[ROTATIONS.index(a) for a in angles]]
        per_view = knn.vectors @ (turned / np.linalg.norm(turned, axis=1, keepdims=True)).T
        sims = per_view.max(axis=1)                                     # as the kNN: best turn per reference
        order = [j for j in np.argsort(-sims) if sims[j] < knn.exclude_self_above]      # leave-one-out
        nearest = order[0]
        same = [j for j in order if knn.labels[j] == (t.document_type, t.issuing_country, t.document_side)]
        record = _fetch(source, t)
        width, height = record.image.size
        x0, y0, x1, y1 = region.bbox
        row = {
            "image": f"{t.transaction_id}:{t.image_id}", "true": label(t), "class_size": sizes[label(t)],
            "predicted": f"{summaries[i].document_type}/{summaries[i].issuing_country}/{summaries[i].document_side}",
            "routed": "review" if summaries[i].needs_review else "AUTO-ACCEPTED",
            "correct": right(i),
            "decision": raws[i].get("decision", ""), "vote_share": raws[i].get("vote_share", ""),
            "margin": raws[i].get("margin", ""), "side_vote": raws[i].get("side_vote", ""),
            "reasons": ",".join(summaries[i].reasons),
            "fallback_full_image": region.region_id.endswith(f":{FULL_IMAGE_INDEX}"),
            "detector_confidence": round(region.confidence, 3),
            "photo_wh": f"{width}x{height}", "box_area_share": round((x1 - x0) * (y1 - y0) / (width * height), 3),
            "box_aspect_w_h": round((x1 - x0) / max(1, y1 - y0), 2),
            "nearest": "/".join(knn.labels[nearest]), "nearest_similarity": round(float(sims[nearest]), 4),
            "turns_compared": "/".join(map(str, angles)), "nearest_turn": int(angles[int(per_view[nearest].argmax())]),
            "nearest_is_true_class": knn.labels[nearest] == (t.document_type, t.issuing_country, t.document_side),
            "best_true_class_similarity": round(float(sims[same[0]]), 4) if same else "none left (1-image class)",
            "rank_of_true_class": (order.index(same[0]) + 1) if same else "",
        }
        rows.append(row)

        boxed = record.image.copy()
        ImageDraw.Draw(boxed).rectangle(region.bbox, outline=(255, 0, 0), width=max(2, width // 150))
        crop = record.image.crop(region.bbox)
        turn = row["nearest_turn"]
        matched = (f'<figure><img src="{_data_uri(crop.rotate(turn, expand=True))}"><figcaption>crop as matched '
                   f'(turned {turn}&deg;)</figcaption></figure>' if turn else "")
        ref_truth, ref_region = truths[nearest], images[nearest].regions[0]
        ref_record = _fetch(source, ref_truth)
        ref_crop = ref_record.image.crop(ref_region.bbox)
        facts = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in row.items())
        if sizes[label(t)] < 2 and not any(c.startswith("<h1 class=single>") for c in cards):   # once
            cards.append("<h1 class=single>Single-sample classes: sent to review by the rule (no reference of "
                         "their own class exists, so the nearest reference is always another class)</h1>")
        cards.append(f"""<section><h2>{html.escape(row['true'])} <small>{html.escape(row['image'])}</small></h2>
<div class="pics"><figure><img src="{_data_uri(boxed)}"><figcaption>photo + detector box</figcaption></figure>
<figure><img src="{_data_uri(crop)}"><figcaption>crop sent to DINOv2</figcaption></figure>{matched}
<figure><img src="{_data_uri(ref_crop)}"><figcaption>nearest reference: {html.escape(row['nearest'])} ({row['nearest_similarity']})</figcaption></figure></div>
<table>{facts}</table></section>""")
        record.image = ref_record.image = None

    with (out / "crops.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Crop review</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa;color:#222}}section{{background:#fff;border:1px solid #ddd;
margin:0 0 16px;padding:12px}}.pics{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0}}img{{max-width:360px;
border:1px solid #ccc}}table{{border-collapse:collapse;margin-top:8px;font-size:13px}}th{{text-align:left;padding:2px 8px;
color:#555}}td{{padding:2px 8px}}</style></head><body><h1>Crop review: {len(rows)} {"failed images where the 1-NN fallback came into play" if args.only_1nn_failures else "documents outside " + majority}</h1>
<p>Real ID images, local only. Experiment <code>{html.escape(args.experiment)}</code>, read from
{html.escape(getattr(source, 'name', 'local files'))}. Routing shown: <code>{html.escape(args.configuration)}</code>, query_rotations <code>{html.escape(knn.query_rotations)}</code>
(single-sample rule {'ON' if sizes_rule else 'off'}).</p>{'<p>Wrong auto-accepts first, then images sent to review.</p>' if args.only_1nn_failures else '<h1>Classes with 2+ images: these should be recognised</h1>'}{''.join(cards)}</body></html>""", encoding="utf-8")
    print(f"{len(rows)} documents -> {out / 'index.html'} (pictures, local only) and {out / 'crops.csv'} (numbers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

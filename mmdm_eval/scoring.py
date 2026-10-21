"""Score model probability files against MMDM, including missing and invalid predictions."""
import argparse
from pathlib import Path

from .common import actor, digest, dump, file_sha, jsonl, rows, save, valid_p
from .download import DATASET_REVISION, resolve_dataset
from .metrics import score, summarize


def score_predictions(data, predictions):
    """Join by question ID; probabilities follow option order or use option-ID keys."""
    by_id = {row["id"]: row for row in data}
    if not data or len(by_id) != len(data):
        raise ValueError("Dataset must contain unique question IDs")
    submitted = {}
    for prediction in predictions:
        rid = prediction.get("id")
        if rid not in by_id:
            raise ValueError(f"Unknown prediction ID: {rid}")
        if rid in submitted:
            raise ValueError(f"Duplicate prediction ID: {rid}")
        submitted[rid] = prediction
    scored = []
    for rid, row in by_id.items():
        supplied = submitted.get(rid)
        record = dict(ok=False, mode="native", status="missing_prediction")
        if supplied is not None:
            if supplied.get("mode", "native") != "native":
                raise ValueError("Grouped runs are scored by mmdm-eval; use that run's summary.json")
            record["status"] = "invalid_prediction"
            if supplied.get("actor_sha256", digest(actor(row))) != digest(actor(row)):
                raise ValueError(f"Prediction input hash mismatch: {rid}")
            if supplied.get("ok", True) is False:
                record["status"] = supplied.get("status", "request_error")
            else:
                try:
                    p = supplied["probabilities"]
                    if isinstance(p, dict):
                        ids = [o["id"] for o in row["options"]]
                        if set(p) != set(ids):
                            raise ValueError("Probability keys must equal option IDs")
                        p = [p[oid] for oid in ids]
                    p = valid_p(p, len(row["options"]))
                    record.update(ok=True, status="complete", probabilities=p)
                except (ValueError, TypeError, KeyError, AssertionError, OverflowError):
                    pass
        scored.append(score(row, record))
    return scored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="local dataset JSONL; default: download pinned MMDM")
    parser.add_argument("--predictions", required=True, help="JSONL with id and probabilities")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--cache-dir")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    args = parser.parse_args(argv)
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error("--output must be empty")
    source = resolve_dataset(args.input, cache_dir=args.cache_dir, revision=args.dataset_revision)
    data = list(rows(source))
    predictions = list(rows(args.predictions))
    values = score_predictions(data, predictions)
    identity = dict(dataset_sha256=file_sha(source), predictions_sha256=file_sha(args.predictions),
                    code={p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    summary = dict(identity=identity, planned=len(data), submitted=len(predictions), overall=summarize(values),
                   categories={c: summarize([v for v in values if v["category"] == c])
                               for c in sorted({v["category"] for v in values})})
    root.mkdir(parents=True, exist_ok=True)
    jsonl(root/"scored.jsonl", values)
    save(root/"summary.json", summary)
    print(dump(summary["overall"]))


if __name__ == "__main__":
    main()

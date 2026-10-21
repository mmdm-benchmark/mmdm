"""Minimal model integration. Replace predict() with your model's inference call."""
import argparse
from pathlib import Path

from mmdm_eval.common import actor, digest, dump, rows, valid_p
from mmdm_eval.download import resolve_dataset


def predict(request):
    # This runnable baseline assigns uniform probability. Only model-visible
    # fields are provided; replace this function to call your model.
    count = len(request["options"])
    return [1 / count] * count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="local dataset JSONL; otherwise auto-download")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = resolve_dataset(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows(source):
            request = actor(row)
            probabilities = valid_p(predict(request), len(request["options"]))
            stream.write(dump(dict(id=row["id"], actor_sha256=digest(request),
                                   probabilities=probabilities)) + "\n")
    print(output)


if __name__ == "__main__":
    main()

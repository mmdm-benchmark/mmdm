"""Download the published MMDM release and tokenizer into the Hugging Face cache."""
import argparse
import json
from pathlib import Path
import re

from .common import file_sha

DATASET = "shenjunhao/mmdm"
DATASET_REVISION = "42a2ff808a22b43960d6fff449b5c5062821c71a"
DATASET_SHA256 = "909e200fd0ac804ac0dafc317d72c7da6736bc43f773957b0fac56e70af30421"
TOKENIZER = "Qwen/Qwen3.5-4B"


def resolve_dataset(path=None, *, cache_dir=None, revision=DATASET_REVISION):
    if path is not None:
        local = Path(path).expanduser().resolve()
        if not local.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {local}")
        return local
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--dataset-revision must be a full Hugging Face commit SHA")
    from huggingface_hub import hf_hub_download
    common = dict(repo_id=DATASET, repo_type="dataset", revision=revision, cache_dir=cache_dir)
    path = Path(hf_hub_download(filename="mmdm.jsonl", **common))
    manifest = json.loads(Path(hf_hub_download(filename="release.json", **common)).read_text())
    expected = manifest["files"]["mmdm.jsonl"]
    if revision == DATASET_REVISION and expected != DATASET_SHA256:
        raise ValueError("Published release manifest does not match the pinned dataset")
    if file_sha(path) != expected:
        raise ValueError("Downloaded dataset checksum mismatch")
    return path


def resolve_tokenizer(value=TOKENIZER, *, cache_dir=None, revision=None):
    local = Path(value).expanduser()
    if local.is_dir():
        local = local / "tokenizer.json"
    if local.is_file():
        return local.resolve()
    if str(value).endswith(".json") or str(value).startswith(("/", ".", "~")):
        raise FileNotFoundError(f"Tokenizer file does not exist: {local}")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=value, filename="tokenizer.json",
                               revision=revision, cache_dir=cache_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--with-tokenizer", action="store_true")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--tokenizer-revision")
    args = parser.parse_args(argv)
    output = {"dataset": str(resolve_dataset(cache_dir=args.cache_dir, revision=args.dataset_revision))}
    if args.with_tokenizer:
        output["tokenizer"] = str(resolve_tokenizer(args.tokenizer, cache_dir=args.cache_dir,
                                                   revision=args.tokenizer_revision))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

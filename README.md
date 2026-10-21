<p align="center"><img src="assets/mmdm-logo.png" width="140" alt="MMDM" /></p>

<h1 align="center">MMDM: Measuring Massive Multitask Decision Making</h1>

<p align="center">
  <a href="https://mmdm-benchmark.github.io">🌐 Homepage</a> ·
  <a href="https://huggingface.co/datasets/shenjunhao/mmdm">🤗 Dataset</a> ·
  <a href="https://mmdm-benchmark.github.io/#leaderboard">🏆 Leaderboard</a>
</p>

The official evaluation toolkit for **MMDM**, a benchmark for text-based System-1 Decision Models. Evaluate single-choice, multiple-correct-option, soft-distribution, and Score tasks with one scoring protocol. See the [homepage](https://mmdm-benchmark.github.io) for the benchmark and results.

## Install

Python 3.11+ on Linux or macOS (Windows: WSL).

```bash
git clone https://github.com/mmdm-benchmark/mmdm.git
cd mmdm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Check the first 20 questions without an API key:

```bash
mmdm-eval --output runs/preflight \
  --context-limit 32768 --service-overhead 1024 \
  --prepare-only --limit 20
```

The toolkit **automatically downloads and caches the dataset and tokenizer**. Data is pinned to a Hugging Face commit and checksum-verified. The default Qwen3.5-4B tokenizer is a proxy for input budgeting; only its tokenizer file is downloaded. Set the context budget for the service you evaluate.

### Evaluate an API

```bash
export JEV_API_KEY="your-key"
bash examples/evaluate_jev.sh
```

For Microsoft Decision-1:

```bash
export OPENROUTER_API_KEY="your-key"
bash examples/evaluate_microsoft.sh
```

Both scripts evaluate the full dataset with 16 concurrent requests. Repeat the same command to resume from the cache. For a short run, append `--limit 20` and use a separate output directory. Override `--model` and `--expected-model` together to evaluate another version.

### Evaluate your model

For a service implementing the typed decision API:

```bash
mmdm-eval --provider custom \
  --endpoint http://127.0.0.1:8000/v1/systemone --model my-model \
  --tokenizer /path/to/tokenizer.json --tokenizer-role provider \
  --context-limit 32768 --service-overhead 1024 \
  --option-cap 1024 --output runs/my-model
```

For any other inference interface, write one prediction per line and run the scorer:

```json
{"id": "question-id", "probabilities": [0.1, 0.7, 0.2]}
```

Probabilities follow the dataset's option order. A mapping from option IDs to probabilities also works. [This runnable example](examples/predict.py) shows where to plug in your model; its default predictor is a uniform baseline.

```bash
python examples/predict.py --output runs/uniform/predictions.jsonl
mmdm-score --predictions runs/uniform/predictions.jsonl --output runs/uniform-score
```

## Data and outputs

Use `--input /path/to/mmdm.jsonl` for a local dataset, or pre-download with `mmdm-download --with-tokenizer`. [Hugging Face](https://huggingface.co/datasets/shenjunhao/mmdm) also provides the dataset in Parquet format.

Each API run writes `summary.json` (overall and per-category scores), `scored.jsonl`, `predictions.jsonl`, and a resumable request cache. `identity.json` records the data, code, tokenizer, model, and context configuration.

- **Single choice:** hard accuracy.
- **Multiple correct options:** exact Top-N set match and probability balance.
- **Soft targets:** Brier score and total variation.
- **Score tasks:** expected-score MAE and normalized MAE.

The evaluator preserves the question, every option, and every criterion. It applies deterministic head–tail truncation to `state` when needed. Larger independent catalogs use hierarchical selection when they exceed the configured option cap. [Evaluation details and API contract →](docs/evaluation.md)

## Leaderboard

🏆 The MMDM leaderboard is available on the [**official homepage**](https://mmdm-benchmark.github.io/#leaderboard).

## License

The evaluation code is released under the [MIT license](LICENSE). Dataset attribution and source terms are provided on [Hugging Face](https://huggingface.co/datasets/shenjunhao/mmdm/blob/main/SOURCE_NOTICES.md).

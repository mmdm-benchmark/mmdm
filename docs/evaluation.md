# Evaluation reference

## Inputs and downloads

`mmdm-eval`, `mmdm-score`, and `mmdm-download` share the published JSONL release. The default revision is pinned in `mmdm_eval/download.py`; its SHA-256 is checked against the release manifest and the package's expected checksum. Downloads use the standard Hugging Face cache. `--cache-dir` selects another cache; `HF_HUB_OFFLINE=1` reuses already-cached assets offline.

Pass `--input FILE.jsonl` to use a local copy. Dataset files stay unchanged. `--tokenizer` accepts a local `tokenizer.json`, its directory, or a Hugging Face model ID; `--tokenizer-revision` pins a model revision. `--dataset-revision` accepts a full dataset commit SHA. A run records the actual data and tokenizer hashes.

`--limit N` selects the first N rows for integration checks. Omit it for the full benchmark. Limited runs have their own identity and output directory.

## API adapters

| Provider | Endpoint | Credential environment variable |
| --- | --- | --- |
| `jev` | `https://api.typesafe.ai/v1/systemone` | `JEV_API_KEY` |
| `microsoft` | `https://openrouter.ai/api/alpha/decisions` | `OPENROUTER_API_KEY` |
| `custom` | `--endpoint` | optional `--api-key-env VARIABLE` |

The built-in model pins are Jev `jev-1.14.0` and Microsoft `microsoft/microsoft-decision-1` (expected response version `microsoft/microsoft-decision-1-20261009`). Set both `--model` and `--expected-model` to change them. For `custom`, the expected response model defaults to `--model`.

Custom endpoints use the same typed request and response shape as [Jev's decision API](https://docs.typesafe.ai/introduction/quickstart):

```json
{
  "model": "my-model",
  "state": "The current situation...",
  "questions": {
    "decision": {
      "type": "choice",
      "instructions": "Which action should be taken?",
      "criteria": {"Action A": "When A applies", "Action B": "When B applies"}
    }
  }
}
```

```json
{
  "model": "my-model",
  "answers": {
    "decision": {"type": "choice", "probabilities": {"Action A": 0.8, "Action B": 0.2}}
  }
}
```

`common.wire` handles the benchmark's Choice, Noul, and Score primitives; `common.parse_native` aligns returned probabilities with the original option order. Noul returns `answers.decision.noul`; Score returns a probability dictionary over string indices `"0"`, `"1"`, etc. Inspect these two functions when implementing an adapter. OpenAI chat-completion endpoints use a different contract; integrate them through the [prediction example](../examples/predict.py) and `mmdm-score`.

Only `state`, `question`, `options`, `task`, and `score_values` enter model requests. Reference answers and category metadata stay in the evaluator.

## Context and candidate catalogs

Declare `--context-limit`, `--service-overhead`, and optionally `--output-reserve`. The compact serialized API request must fit within:

```text
available = context_limit - service_overhead - output_reserve
```

The default tokenizer is Qwen3.5-4B, recorded as a **proxy**. For a service's own tokenizer, use `--tokenizer PATH_OR_MODEL --tokenizer-role provider`. The example scripts use a 32,768-token evaluation budget and a 1,024-token service allowance; these are run settings. Adjust them for your endpoint.

The context adapter preserves all fixed input fields, then retains approximately equal token-boundary portions from the start and end of the original Unicode `state`, with an omission marker. It recounts the complete request. If the fixed fields alone exceed the budget, the result is `fixed_fields_exceed_budget`; if the provider rejects the input length, it is `provider_context_capacity`. Both appear in coverage and failure counts.

For independent catalog tasks above `--option-cap`, the evaluator partitions options deterministically, keeps the top N from each group, and repeats until a final group fits. N is one for single choice and the reference set size for multiple-correct-option evaluation. The API receives neither gold IDs nor N. Every original option participates; all groups share the same budgeted state. Cross-option logic and soft-distribution tasks require a native full-catalog response. The complete catalog is budgeted before grouping.

Native and grouped results are reported separately. Grouped final probabilities are conditional on the surviving set, so full-catalog Brier and Score MAE are computed only for native responses.

## Scoring

- **Hard single accuracy:** the unique highest-probability option must be correct. A tie for first place is incorrect.
- **Hard-set accuracy:** all gold members must rank strictly above all nonmembers. N is the number of gold members; a tie across the boundary is incorrect.
- **Top-N balance:** probabilities within the predicted top N are renormalized and compared with a uniform distribution. One means equal allocation; zero means all top-N mass is on one member. Balance and set accuracy are separate metrics.
- **Soft Brier:** sum of squared errors over options, averaged across valid questions. Lower is better.
- **Soft total variation:** half the sum of absolute probability errors.
- **Score MAE:** absolute difference between predicted and reference expected numeric scores. Normalized MAE divides by the score-scale range.

Hard `accuracy_all` includes failed or missing predictions as incorrect; `accuracy_valid` uses valid responses. Distribution metrics include their valid counts. Report coverage with the scores. Probabilities must be finite, nonnegative, at most one, and sum to one within 0.02; accepted rounding drift is renormalized.

## Offline predictions

`mmdm-score` joins predictions by `id`, counts missing and malformed probabilities as invalid, and rejects duplicate or unknown IDs. Probabilities may be an ordered array or a complete dictionary keyed by option ID. Optional `actor_sha256` binds a prediction to its input. A failed request can be recorded as `{"id": "...", "ok": false, "status": "request_error"}`.

The offline scorer accepts native full-distribution predictions. Grouped API runs are scored directly by `mmdm-eval`, which includes their selection trace in `predictions.jsonl`.

## Resume and files

Repeat a command with the same output directory to resume. Successful predictions and HTTP responses are cached; transient failures are retried. A model, dataset, code, tokenizer, budget, seed, or selection change requires a new output directory.

| File | Contents |
| --- | --- |
| `identity.json` | Dataset/code/tokenizer hashes and run configuration |
| `predictions.jsonl` | Probabilities or grouping traces, context accounting and API usage |
| `calls.sqlite` | Request and response cache, excluding authorization headers |
| `scored.jsonl` | Per-question metrics |
| `summary.json` | Overall, category and context-view aggregates |
| `status.json` | Progress, valid count, attempts and cache hits |
| `COMPLETE.json` / `BLOCKED.json` | Run outcome |

`--prepare-only` writes `context-audit.jsonl` and `context-summary.json` without inference calls. Use a separate output directory for the subsequent evaluation. Authentication or quota errors stop the run with a nonzero exit status; resume after resolving them.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover label isolation, Unicode truncation, protected fields, grouping, resume, download checksums, score denominators, and API integration with a local fixture service.

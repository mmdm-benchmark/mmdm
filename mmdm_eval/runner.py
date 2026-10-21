"""MMDM native/grouped evaluation with an immutable dataset and budgeted views."""
import argparse
import asyncio
from collections import Counter
import fcntl
from pathlib import Path
import time

from .common import actor, digest, dump, file_sha, json, jsonl, parse_native, rows, save, wire
from .client import ApiError, Client, PROVIDERS, ProviderBlocked
from .context import Budget, FixedFieldsTooLong, JsonTokenizerCounter, PROTOCOL, adapt_state
from .grouped_choice import ContextCapacityError, score_selection, select_hierarchical
from .metrics import label_kind, score, summarize
from .download import DATASET_REVISION, TOKENIZER, resolve_dataset, resolve_tokenizer
from urllib.parse import urlparse

# These catalog tasks were checked for independent option semantics. Cross-option
# logic and full soft distributions cannot use a hierarchical selection adapter.
GROUPABLE = {
    ("mmdm_v2_revision_workflow", "skill_composition"),
    ("mmdm_v2_practical_plans", "plan_comparison"),
    ("toolret", "tool_relevance"), ("skillsbench", "skill_composition"),
    ("skillsbench", "skill_bundle_selection"),
    ("skilldag", "skill_selection"),
    ("more_skillbank_review_v2_routing", "skill_library_routing"),
}


def request_counter(tokenizer, model, role):
    return JsonTokenizerCounter(tokenizer, lambda value: wire(value, model)[0], tokenizer_role=role)


def append(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(dump(record)+"\n")
        stream.flush()


async def evaluate_row(row, counter, budget, client, config, *, cap, seed):
    record = dict(id=row["id"], actor_sha256=digest(actor(row)), mode="native", ok=False)
    start = time.monotonic()
    try:
        effective, context = adapt_state(row, counter, budget)
        record["context"] = context
        if len(effective["options"]) > cap:
            record["mode"] = "grouped"
            if ((row.get("source", {}).get("dataset"), row.get("task_subcategory")) not in GROUPABLE
                    or row.get("task_category") == "option_logic"
                    or row["task"] not in {"choice", "multichoice"}
                    or label_kind(row) == "soft"):
                record["status"] = "unsupported_grouping"
                return record

        async def query(subset, path):
            # All groups share one precomputed state. No per-group truncation.
            if subset["state"] != effective["state"]:
                raise AssertionError("group state changed")
            count = counter(subset)
            if count > budget.available:
                raise ContextCapacityError("group_request_exceeds_budget")
            body, keys = wire(subset, config["model"])
            raw, key = await client.call(body)
            if raw.get("model") != config["expected_model"]:
                raise ValueError("model_version_changed")
            p = parse_native(body, keys, raw)
            return p, dict(cache_key=key, request_sha256=digest(body), model=raw["model"],
                           usage=raw.get("usage", {}), input_tokens=count,
                           effective_state_sha256=context["effective_state_sha256"])

        if record["mode"] == "native":
            p, meta = await query(effective, "native")
            record.update(probabilities=p, **meta)
        else:
            k = len(row["correct_option_ids"]) if label_kind(row) == "hard_set" else 1
            if not 1 <= k < cap:
                record["status"] = "unsupported_grouping"
                return record
            result = await select_hierarchical(effective, query, cap=cap, k=k, seed=seed)
            record.update(result=result, metrics=score_selection(row, result))
        record.update(ok=True, status="complete")
    except FixedFieldsTooLong as exc:
        record.update(status="fixed_fields_exceed_budget", context=exc.metadata)
    except ProviderBlocked:
        raise
    except ApiError as exc:
        # A provider rejection remains visible. No answer-dependent shortening or
        # retry-driven budget changes are used to select a favorable view.
        record.update(status="provider_context_capacity" if exc.context else "request_error",
                      error=str(exc), cache_key=exc.cache_key, http_status=exc.status)
    except ContextCapacityError as exc:
        record.update(status="context_capacity", error=str(exc))
    except (ValueError, KeyError, TypeError, AssertionError) as exc:
        record.update(status="protocol_error", error=type(exc).__name__+": "+str(exc)[:200])
    finally:
        record["wall_s"] = time.monotonic()-start
    return record


def report(data, predictions, identity):
    values = [score(r, predictions[r["id"]]) for r in data if r["id"] in predictions]
    context_groups = {}
    for name, predicate in (("full_input", lambda c: c.get("truncated") is False),
                            ("truncated_state", lambda c: c.get("truncated") is True),
                            ("fixed_fields_exceed_budget", lambda c: c.get("status") == "fixed_fields_exceed_budget")):
        context_groups[name] = summarize([v for v in values if predicate(predictions[v["id"]].get("context", {}))])
    return dict(identity=identity, planned=len(data), processed=len(values), pending=len(data)-len(values),
                overall=summarize(values), by_context=context_groups,
                context_statuses=dict(Counter(p.get("context", {}).get("status", "unknown") for p in predictions.values())),
                categories={c: summarize([v for v in values if v["category"] == c])
                            for c in sorted({v["category"] for v in values})})


async def run(args, *, counter_override=None, client_factory=Client):
    source = resolve_dataset(args.input, cache_dir=getattr(args, "cache_dir", None),
                             revision=getattr(args, "dataset_revision", DATASET_REVISION))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"evaluate.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = None
    try:
        data = list(rows(source))
        if not data or len({r["id"] for r in data}) != len(data):
            raise ValueError("input must have unique IDs and at least one row")
        if getattr(args, "limit", None):
            data = data[:args.limit]
        config = dict(PROVIDERS[args.provider])
        if args.model: config["model"] = args.model
        if args.expected_model: config["expected_model"] = args.expected_model
        if args.provider == "custom":
            config.update(url=args.endpoint, key_env=args.api_key_env,
                          expected_model=args.expected_model or args.model)
        budget = Budget(args.context_limit, args.service_overhead, args.output_reserve)
        counter = counter_override or request_counter(resolve_tokenizer(
            args.tokenizer, cache_dir=getattr(args, "cache_dir", None),
            revision=getattr(args, "tokenizer_revision", None)), config["model"], args.tokenizer_role)
        identity = dict(protocol=PROTOCOL, dataset_sha256=file_sha(source), provider=args.provider,
                        model=config["model"], expected_model=config["expected_model"], endpoint=config["url"],
                        budget=budget.__dict__, counter=counter.identity, seed=args.seed, option_cap=args.option_cap,
                        mode="prepare" if args.prepare_only else "evaluate",
                        limit=getattr(args, "limit", None), selected_ids_sha256=digest([r["id"] for r in data]),
                        code={p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        identity_path = root/"identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("resume_identity_mismatch: use a new output directory")
        else:
            if any(p.name != "evaluate.lock" for p in root.iterdir()):
                raise ValueError("new run output directory must be empty")
            save(identity_path, identity)
        pp = root/"predictions.jsonl"
        predictions = {r["id"]: r for r in rows(pp)} if pp.exists() else {}
        byid = {r["id"]: r for r in data}
        for rid, rec in predictions.items():
            if rid not in byid or rec["actor_sha256"] != digest(actor(byid[rid])):
                raise ValueError("prediction_actor_mismatch")

        if args.prepare_only:
            prepared = []
            for r in data:
                try:
                    _, meta = adapt_state(r, counter, budget)
                except FixedFieldsTooLong as exc:
                    meta = exc.metadata
                prepared.append(dict(id=r["id"], context=meta))
            jsonl(root/"context-audit.jsonl", prepared)
            save(root/"context-summary.json", dict(rows=len(data), identity=identity,
                 truncated=sum(p["context"].get("truncated", False) for p in prepared),
                 fixed_fields_exceed_budget=sum(p["context"]["status"] == "fixed_fields_exceed_budget" for p in prepared)))
            if file_sha(source) != identity["dataset_sha256"]:
                raise ValueError("source_changed_during_preparation")
            print(dump(dict(output=str(root), rows=len(data), status="prepared")))
            return

        client = client_factory(root, args.provider, args.concurrency, digest(identity), config=config)
        queue = asyncio.Queue()
        for row in data:
            previous = predictions.get(row["id"])
            if previous is None or previous["status"] == "request_error":
                queue.put_nowait(row)
        fatal = None

        def snapshot(status):
            save(root/"status.json", dict(status=status, planned=len(data), processed=len(predictions),
                 valid=sum(p["ok"] for p in predictions.values()), updated_at=time.time(),
                 http_attempts_this_process=client.attempts, cache_hits_this_process=client.cache_hits))

        async def worker():
            nonlocal fatal
            while fatal is None:
                try: row = queue.get_nowait()
                except asyncio.QueueEmpty: return
                try:
                    rec = await evaluate_row(row, counter, budget, client, config,
                                             cap=args.option_cap, seed=args.seed)
                except ProviderBlocked as exc:
                    fatal = str(exc)
                    return
                predictions[row["id"]] = rec
                append(pp, rec)
                if len(predictions) % 25 == 0 or queue.empty(): snapshot("running")

        (root/"COMPLETE.json").unlink(missing_ok=True)
        snapshot("running")
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        if file_sha(source) != identity["dataset_sha256"]:
            raise ValueError("source_changed_during_evaluation")
        stats = report(data, predictions, identity)
        save(root/"summary.json", stats)
        jsonl(root/"scored.jsonl", [score(r, predictions[r["id"]]) for r in data if r["id"] in predictions])
        snapshot("blocked" if fatal else "complete")
        if fatal:
            save(root/"BLOCKED.json", dict(reason=fatal, pending=len(data)-len(predictions)))
            raise RuntimeError("Provider blocked the run; see BLOCKED.json. Resume with the same command after resolving it.")
        else:
            if len(predictions) != len(data): raise AssertionError("missing_predictions")
            (root/"BLOCKED.json").unlink(missing_ok=True)
            save(root/"COMPLETE.json", dict(rows=len(data), valid=stats["overall"]["valid"],
                 dataset_sha256=identity["dataset_sha256"], summary_sha256=file_sha(root/"summary.json")))
        print(dump(stats["overall"]))
    finally:
        if client is not None: client.close()
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="local JSONL; omitted: download the pinned MMDM release")
    parser.add_argument("--cache-dir", help="Hugging Face download cache")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--output", "--root", dest="output", required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="jev")
    parser.add_argument("--model")
    parser.add_argument("--expected-model")
    parser.add_argument("--endpoint", help="full typed decision API URL for --provider custom")
    parser.add_argument("--api-key-env", help="environment variable for a custom endpoint's bearer token")
    parser.add_argument("--tokenizer", default=TOKENIZER, help="tokenizer.json path or Hugging Face model ID")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--tokenizer-role", choices=["provider", "proxy"], default="proxy")
    parser.add_argument("--context-limit", type=int, required=True)
    parser.add_argument("--service-overhead", type=int, required=True)
    parser.add_argument("--output-reserve", type=int, default=0)
    parser.add_argument("--option-cap", type=int, default=255)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--prepare-only", action="store_true", help="no API calls or API key needed")
    parser.add_argument("--limit", type=int, help="first N rows for a quick check; omit for a full evaluation")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.option_cap < 2: parser.error("invalid concurrency or option cap")
    if args.limit is not None and args.limit < 1: parser.error("--limit must be positive")
    if args.provider == "custom":
        if not args.endpoint or not args.model:
            parser.error("--provider custom requires --endpoint and --model")
        url = urlparse(args.endpoint)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            parser.error("--endpoint must be an HTTP(S) URL without embedded credentials")
    elif args.endpoint or args.api_key_env:
        parser.error("--endpoint and --api-key-env apply to --provider custom")
    asyncio.run(run(args))

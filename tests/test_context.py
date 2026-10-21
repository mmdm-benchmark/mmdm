import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import unittest

from mmdm_eval.common import actor, digest, wire
from mmdm_eval.client import ApiError, PROVIDERS, is_context_error
from mmdm_eval.context import Budget, FixedFieldsTooLong, MARKER, adapt_state, JsonTokenizerCounter
from mmdm_eval.metrics import score, summarize
from mmdm_eval.runner import evaluate_row, report, run


class CharacterCounter:
    """Exact fixture counter, deliberately not presented as a model tokenizer."""
    identity = {"fixture": "unicode-codepoints-compact-wire-json"}

    def __call__(self, row):
        return len(json.dumps(wire(row, "fixture")[0], ensure_ascii=False, separators=(",", ":")))

    def text_tokens(self, text): return len(text)
    def boundaries(self, text): return list(range(len(text)+1))


def row(task="choice", count=4):
    r = dict(id="fixture", state="头部🙂 HEAD " + "middle evidence 中间 "*100 + " TAIL 尾部🦊",
             question="Which option follows the exceptions? Keep this question.", task=task,
             options=[dict(id=str(i), name="Action "+str(i), criteria="Use this only if option 0 fails; detail "+str(i)) for i in range(count)],
             gold=[1.0]+[0.0]*(count-1), correct_option_ids=["0"],
             task_category="tool_skill_selection", task_subcategory="plan_comparison",
             source={"dataset": "mmdm_v2_practical_plans"})
    if task == "score": r["score_values"] = list(range(count))
    if task == "noul":
        r["options"] = [dict(id="f", name="false", criteria="negative conditions"), dict(id="t", name="true", criteria="positive conditions")]
        r["gold"] = [1., 0.]
    if task == "multichoice":
        r["label_semantics"] = "set_hard"
        r["gold"] = [.5, .5]+[0.]*(count-2)
        r["correct_option_ids"] = ["0", "1"]
    return r


class FakeClient:
    """Exercises real wire, parser, metrics, persistence and resume without APIs."""
    def __init__(self, *args, **kwargs):
        self.attempts = 0
        self.cache_hits = 0
        self.bodies = []
    async def call(self, body):
        self.attempts += 1
        self.bodies.append(deepcopy(body))
        q = body["questions"]["decision"]
        if q["type"] == "noul": answer = {"type": "noul", "noul": .3}
        else:
            criteria = q["criteria"]
            keys = list(criteria) if isinstance(criteria, dict) else list(map(str, range(len(criteria))))
            weights = list(range(len(keys), 0, -1))
            answer = dict(type=q["type"], probabilities={k: w/sum(weights) for k, w in zip(keys, weights)})
        return {"model": PROVIDERS["jev"]["expected_model"], "answers": {"decision": answer}}, digest(body)
    def close(self): pass


class ContextTests(unittest.TestCase):
    def setUp(self): self.counter = CharacterCounter()

    def test_small_input_unchanged_at_exact_limit(self):
        r = row(); before = deepcopy(r)
        budget = Budget(self.counter(actor(r))+17, 13, 4)
        view, meta = adapt_state(r, self.counter, budget)
        self.assertEqual(view, actor(r)); self.assertEqual(r, before)
        self.assertFalse(meta["truncated"])
        self.assertEqual(meta["input_tokens_with_reserves"], budget.context_limit)
        self.assertNotIn("gold", view)

    def test_all_primitives_protected_and_deterministic(self):
        for kind in ["choice", "noul", "score", "multichoice"]:
            r = row(kind); before = deepcopy(r)
            fixed = self.counter({**actor(r), "state": ""})
            budget = Budget(fixed+150, 19, 11)
            view, meta = adapt_state(r, self.counter, budget)
            self.assertEqual((view, meta), adapt_state(r, self.counter, budget))
            self.assertEqual({k: v for k, v in view.items() if k != "state"}, {k: v for k, v in actor(r).items() if k != "state"})
            self.assertEqual(r, before)
            self.assertTrue(view["state"].startswith("头部🙂 HEAD"))
            self.assertTrue(view["state"].endswith("TAIL 尾部🦊"))
            self.assertIn(MARKER, view["state"])
            self.assertLessEqual(meta["input_tokens_with_reserves"], budget.context_limit)
            self.assertNotIn("\ufffd", view["state"])
            reconstructed = MARKER.join(r["state"][a:b] for a, b in meta["kept_original_char_ranges"])
            self.assertEqual(reconstructed, view["state"])

    def test_fixed_fields_failure_does_not_drop_options(self):
        r = row(count=1024); before = deepcopy(r)
        fixed = self.counter({**actor(r), "state": ""})
        with self.assertRaises(FixedFieldsTooLong) as hit:
            adapt_state(r, self.counter, Budget(fixed-1, 0))
        self.assertEqual(hit.exception.metadata["status"], "fixed_fields_exceed_budget")
        self.assertEqual(r, before)

    def test_no_room_for_marker_can_empty_state(self):
        r = row()
        fixed = self.counter({**actor(r), "state": ""})
        view, meta = adapt_state(r, self.counter, Budget(fixed, 0))
        self.assertEqual(view["state"], "")
        self.assertFalse(meta["marker_used"])
        self.assertTrue(meta["truncated"])

    def test_larger_window_retains_more_original_text(self):
        r = row()
        fixed = self.counter({**actor(r), "state": ""})
        retained = [adapt_state(r, self.counter, Budget(fixed+b, 10))[1]["retained_original_chars"] for b in [80, 160, 320, 10000]]
        self.assertEqual(retained, sorted(retained)); self.assertEqual(retained[-1], len(r["state"]))

    def test_gold_has_no_effect_on_visible_view(self):
        a = row(); b = deepcopy(a); b["gold"] = [0., 0., 0., 1.]; b["correct_option_ids"] = ["3"]
        self.assertEqual(adapt_state(a, self.counter, Budget(900, 30)), adapt_state(b, self.counter, Budget(900, 30)))

    def test_budget_validation(self):
        for args in [(0, 0), (5, 5), (10, -1), (True, 0), (10, 3, 8)]:
            with self.assertRaises(ValueError): Budget(*args)

    def test_true_full_request_count_not_sum_of_fields(self):
        class EscapedCounter(CharacterCounter):
            def __call__(self, r):
                return super().__call__(r) + r["state"].count("x")*7
        r = row(); r["state"] = "x"*300
        counter = EscapedCounter()
        view, meta = adapt_state(r, counter, Budget(1000, 16))
        self.assertLessEqual(counter(view)+16, 1000)
        self.assertEqual(meta["input_tokens"], counter(view))

    def test_mock_transport_preserves_fields_and_scoring(self):
        for kind in ["choice", "noul", "score", "multichoice"]:
            r = row(kind); client = FakeClient()
            rec = asyncio.run(evaluate_row(r, self.counter, Budget(900, 16), client, PROVIDERS["jev"], cap=255, seed=7))
            self.assertTrue(rec["ok"], rec)
            expected, _ = wire(r, PROVIDERS["jev"]["model"])
            self.assertEqual(client.bodies[0]["questions"], expected["questions"])
            self.assertNotEqual(client.bodies[0]["state"], r["state"])
            self.assertTrue(score(r, rec)["valid"])

    def test_grouped_catalog_keeps_every_option_and_same_state(self):
        r = row(count=32); client = FakeClient()
        fixed = self.counter({**actor(r), "state": ""})
        rec = asyncio.run(evaluate_row(r, self.counter, Budget(fixed+200, 16), client, PROVIDERS["jev"], cap=8, seed=7))
        self.assertTrue(rec["ok"], rec)
        seen = set()
        for body in client.bodies:
            seen.update(body["questions"]["decision"]["criteria"])
        self.assertEqual(seen, {o["name"] for o in r["options"]})
        self.assertEqual(len({body["state"] for body in client.bodies}), 1)
        self.assertTrue(rec["context"]["truncated"])

    def test_capacity_rejection_stays_explicit(self):
        class Reject(FakeClient):
            async def call(self, body): raise ApiError(400, "cache-key", context=True)
        r = row()
        rec = asyncio.run(evaluate_row(r, self.counter, Budget(900, 16), Reject(), PROVIDERS["jev"], cap=255, seed=7))
        self.assertEqual(rec["status"], "provider_context_capacity")
        self.assertFalse(rec["ok"])
        self.assertTrue(is_context_error(400, {"error": "maximum context length exceeded"}))
        self.assertFalse(is_context_error(402, {"error": "quota exhausted"}))

    def test_resume_identity_and_frozen_input(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base/"data.jsonl"
            r = row(); source.write_text(json.dumps(r, ensure_ascii=False)+"\n")
            original = source.read_bytes()
            args = types.SimpleNamespace(input=source, output=base/"run", provider="jev", model=None, expected_model=None,
                context_limit=900, service_overhead=16, output_reserve=0, tokenizer=None, tokenizer_role="proxy",
                prepare_only=False, concurrency=2, seed=7, option_cap=255)
            asyncio.run(run(args, counter_override=self.counter, client_factory=FakeClient))
            pp = args.output/"predictions.jsonl"; first = pp.read_bytes()
            asyncio.run(run(args, counter_override=self.counter, client_factory=FakeClient))
            self.assertEqual(first, pp.read_bytes()); self.assertEqual(original, source.read_bytes())
            summary = json.loads((args.output/"summary.json").read_text())
            self.assertEqual(summary["by_context"]["truncated_state"]["rows"], 1)
            self.assertEqual(summary["pending"], 0)
            args.context_limit += 1
            with self.assertRaisesRegex(ValueError, "resume_identity_mismatch"):
                asyncio.run(run(args, counter_override=self.counter, client_factory=FakeClient))

    def test_real_tokenizer_file_when_available(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError:
            self.skipTest("optional tokenizers runtime is not installed")
        with tempfile.TemporaryDirectory() as directory:
            vocab = {c: i for i, c in enumerate(["[UNK]", *pre_tokenizers.ByteLevel.alphabet()])}
            tok = Tokenizer(models.BPE(vocab, [], unk_token="[UNK]"))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.enable_truncation(10); tok.enable_padding(length=20)
            path = Path(directory)/"tokenizer.json"; tok.save(str(path))
            counter = JsonTokenizerCounter(path, lambda r: wire(r, "fixture")[0], tokenizer_role="proxy")
            r = row(); fixed = counter({**actor(r), "state": ""})
            view, meta = adapt_state(r, counter, Budget(fixed+200, 10))
            self.assertGreater(meta["original_input_tokens"], 20)
            self.assertLessEqual(counter(view)+10, fixed+200)
            self.assertIn("头部🙂", view["state"])
            self.assertTrue(view["state"].endswith("尾部🦊"))


if __name__ == "__main__": unittest.main()

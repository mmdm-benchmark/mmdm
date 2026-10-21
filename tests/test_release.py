import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from mmdm_eval.common import file_sha
from mmdm_eval.download import DATASET_REVISION, resolve_dataset, resolve_tokenizer
from mmdm_eval.metrics import summarize
from mmdm_eval.runner import run
from mmdm_eval.scoring import score_predictions
from test_context import CharacterCounter, row


class ScoringTests(unittest.TestCase):
    def test_missing_invalid_and_wrong_answers_keep_full_denominator(self):
        data = []
        for i in range(4):
            item = row(); item["id"] = str(i); data.append(item)
        predictions = [dict(id="0", probabilities={"0": .8, "1": .1, "2": .05, "3": .05}),
                       dict(id="1", probabilities=[0, 1, 0, 0]),
                       dict(id="2", probabilities=[float("nan"), 0, 0, 1])]
        values = score_predictions(data, predictions)
        summary = summarize(values)
        self.assertEqual(summary["hard_single"]["accuracy_all"], .25)
        self.assertEqual(summary["hard_single"]["accuracy_valid"], .5)
        self.assertEqual(summary["coverage"], .5)
        self.assertEqual(values[3]["status"], "missing_prediction")

    def test_soft_score_and_set_metrics(self):
        soft = row("score"); soft["id"] = "soft"; soft["gold"] = [.1, .2, .3, .4]
        hard_set = row("multichoice"); hard_set["id"] = "set"
        values = score_predictions([soft, hard_set], [
            dict(id="soft", probabilities=[.1, .2, .3, .4]),
            dict(id="set", probabilities=[.4, .4, .1, .1])])
        self.assertEqual(values[0]["brier"], 0)
        self.assertEqual(values[0]["score_mae"], 0)
        self.assertEqual(values[1]["correct"], 1)
        self.assertEqual(values[1]["native_set_balance"], 1)
        tie = score_predictions([hard_set], [dict(id="set", probabilities=[.4, .2, .2, .2])])
        self.assertEqual(tie[0]["correct"], 0)

    def test_unknown_duplicate_or_changed_input_rejected(self):
        r = row()
        for records in [[dict(id="unknown")], [dict(id=r["id"])]*2,
                        [dict(id=r["id"], actor_sha256="wrong")]]:
            with self.assertRaises(ValueError): score_predictions([r], records)


class DownloadTests(unittest.TestCase):
    def test_local_paths_need_no_network_and_do_not_change_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"local.jsonl"; data.write_text('{}\n')
            tok = root/"tokenizer.json"; tok.write_text('{}')
            with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected download")):
                self.assertEqual(resolve_dataset(data), data.resolve())
                self.assertEqual(resolve_tokenizer(root), tok.resolve())
            self.assertEqual(data.read_text(), '{}\n')
            with self.assertRaises(FileNotFoundError): resolve_dataset(root/"absent.jsonl")

    def test_download_uses_one_revision_and_validates_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"mmdm.jsonl"; data.write_text('{}\n')
            manifest = root/"release.json"
            manifest.write_text(json.dumps({"files": {"mmdm.jsonl": file_sha(data)}}))
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return str(root/kwargs["filename"])
            with patch("huggingface_hub.hf_hub_download", side_effect=download):
                self.assertEqual(resolve_dataset(revision="a"*40), data)
                self.assertTrue(all(c["revision"] == "a"*40 and c["repo_type"] == "dataset" for c in calls))
                data.write_text('changed')
                with self.assertRaisesRegex(ValueError, "checksum"): resolve_dataset(revision="a"*40)
                with self.assertRaisesRegex(ValueError, "full Hugging Face commit"): resolve_dataset(revision="main")


class HttpIntegrationTests(unittest.TestCase):
    def test_real_transport_resume_and_gold_isolation(self):
        bodies = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                keys = list(body["questions"]["decision"]["criteria"])
                payload = json.dumps(dict(model="test-model", answers=dict(decision=dict(
                    type="choice", probabilities={k: 1/len(keys) for k in keys})))).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root/"data.jsonl"
                r = row(); source.write_text(json.dumps(r)+'\n')
                args = types.SimpleNamespace(input=source, output=root/"run", provider="custom",
                    model="test-model", expected_model="test-model", api_key_env=None,
                    endpoint=f"http://127.0.0.1:{server.server_port}/decide",
                    context_limit=900, service_overhead=16, output_reserve=0, tokenizer=None,
                    tokenizer_role="proxy", prepare_only=False, concurrency=2, seed=7, option_cap=255)
                with patch.dict("os.environ", {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
                    asyncio.run(run(args, counter_override=CharacterCounter()))
                    asyncio.run(run(args, counter_override=CharacterCounter()))
                self.assertEqual(len(bodies), 1)
                self.assertEqual(set(bodies[0]), {"state", "questions", "model"})
                self.assertNotIn("gold", json.dumps(bodies[0]))
                result = json.loads((root/"run/summary.json").read_text())
                self.assertEqual(result["overall"]["valid"], 1)
                self.assertEqual(result["overall"]["hard_single"]["accuracy_all"], 0)
                self.assertTrue((root/"run/COMPLETE.json").exists())
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()

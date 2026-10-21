"""Resumable decision API transport; keys are read only from the environment."""
import asyncio
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request

from .common import digest, dump

PROVIDERS = {
    "jev": {"url": "https://api.typesafe.ai/v1/systemone", "model": "jev-1.14.0",
            "expected_model": "jev-1.14.0", "key_env": "JEV_API_KEY"},
    "microsoft": {"url": "https://openrouter.ai/api/alpha/decisions",
                  "model": "microsoft/microsoft-decision-1",
                  "expected_model": "microsoft/microsoft-decision-1-20261009",
                  "key_env": "OPENROUTER_API_KEY"},
    "custom": {"url": None, "model": None, "expected_model": None, "key_env": None},
}


class ApiError(RuntimeError):
    def __init__(self, status, key, context=False):
        self.status, self.cache_key, self.context = status, key, context
        super().__init__(f"http_{status}_cache_{key}")


class ProviderBlocked(ApiError):
    pass


def is_context_error(status, response):
    text = dump(response).lower()
    return status == 413 or (status in (400, 422) and any(term in text for term in (
        "max_tokens_exceeded", "context limit", "context length", "maximum context",
        "too many tokens", "limit for the state plus all questions")))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, root, provider, concurrency, run_identity, *, config=None):
        self.config = config or PROVIDERS[provider]
        key_env = self.config["key_env"]
        self.secret = os.environ.get(key_env, "").strip() if key_env else ""
        if key_env and not self.secret:
            raise ValueError("set " + self.config["key_env"])
        self.root = root
        self.identity = run_identity
        self.sem = asyncio.Semaphore(concurrency)
        self.db = sqlite3.connect(root/"calls.sqlite")
        self.db.execute("pragma journal_mode=WAL")
        self.db.execute("create table if not exists calls(key text primary key, request text, response text, status integer, at real)")
        self.db.commit()
        self.attempts = 0
        self.cache_hits = 0
        self.fatal = None

    def post(self, body):
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["Authorization"] = "Bearer " + self.secret
        request = urllib.request.Request(
            self.config["url"], data=json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(),
            headers=headers)
        opener = urllib.request.build_opener(NoRedirect())
        try:
            with opener.open(request, timeout=360) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"error_type": type(exc).__name__}
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                payload = {"error_type": "non_object_response"}
        except (ValueError, UnicodeError):
            payload = {"error_type": "non_json_response"}
        return status, payload

    async def call(self, body):
        key = digest({"run_identity": self.identity, "endpoint": self.config["url"], "body": body})
        old = self.db.execute("select response,status from calls where key=?", (key,)).fetchone()
        if old:
            response, status = json.loads(old[0]), old[1]
            if status == 200 and not response.get("error") and not response.get("error_type"):
                self.cache_hits += 1
                return response, key
            if is_context_error(status, response):
                raise ApiError(status, key, context=True)
        for attempt in range(5):
            async with self.sem:
                if self.fatal:
                    raise self.fatal
                status, response = await asyncio.to_thread(self.post, body)
                self.attempts += 1
                self.db.execute("insert or replace into calls values(?,?,?,?,?)",
                                (key, dump(body), dump(response), status, time.time()))
                self.db.commit()
                with (self.root/"http-attempts.jsonl").open("a") as f:
                    f.write(dump(dict(cache_key=key, status=status, at=time.time()))+"\n")
                if status == 200 and not response.get("error") and not response.get("error_type"):
                    return response, key
                if status in (401, 402, 403, 404) or 300 <= status < 400:
                    self.fatal = ProviderBlocked(status, key)
                    raise self.fatal
                if status not in (0, 408, 409, 425, 429, 500, 502, 503, 504, 529):
                    raise ApiError(status, key, context=is_context_error(status, response))
            if attempt < 4:
                await asyncio.sleep(min(60, 2**attempt))
        raise ApiError(status, key)

    def close(self):
        self.db.close()

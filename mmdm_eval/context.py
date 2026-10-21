"""Gold-blind, deterministic state-only context adaptation.

The counter measures the complete provider request, including fixed fields.
Service and output allowances are explicit run parameters, never inferred from
the benchmark's stored length estimates. Original strings are sliced by Python
character offsets; token decoding cannot corrupt a partial UTF-8 character.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

from .common import actor, digest, file_sha

PROTOCOL = "mmdm-state-head-tail-v1"
MARKER = "\n\n[... middle of state omitted ...]\n\n"


class FixedFieldsTooLong(ValueError):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__("question/options/criteria and overhead exceed context budget")


@dataclass(frozen=True)
class Budget:
    context_limit: int
    service_overhead: int
    output_reserve: int = 0

    def __post_init__(self):
        values = asdict(self)
        if any(type(v) is not int for v in values.values()):
            raise ValueError("context budgets must be integers")
        if self.context_limit < 1 or self.service_overhead < 0 or self.output_reserve < 0:
            raise ValueError("invalid context budget")
        if self.available < 1:
            raise ValueError("service/output reserves exhaust context budget")

    @property
    def available(self):
        return self.context_limit - self.service_overhead - self.output_reserve


class JsonTokenizerCounter:
    """Local estimate of compact wire JSON; provider hidden tokens are reserved.

Use the provider tokenizer when available. If a proxy tokenizer is used, declare
it in the run and supply an appropriate overhead. Unknown provider capacity is
never silently replaced with 65,535 or a character-based approximation.
"""
    def __init__(self, tokenizer_path, render, *, tokenizer_role):
        from tokenizers import Tokenizer
        import tokenizers
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # A tokenizer.json may itself contain truncation or padding settings.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.render = render
        self.identity = dict(tokenizer_sha256=file_sha(tokenizer_path),
                             tokenizer_role=tokenizer_role,
                             tokenizers_version=tokenizers.__version__,
                             serialization="compact-wire-json-utf8-v1",
                             special_tokens=False)

    def text_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def boundaries(self, text):
        # Offset sets keep slices on original-text boundaries, including Unicode.
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return sorted({0, len(text), *(end for start, end in encoding.offsets if end > start)})

    def __call__(self, value):
        import json
        body = self.render(value)
        text = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return self.text_tokens(text)


def adapt_state(row, counter, budget):
    """Return a separate actor view plus complete accounting, without labels.

Always recount the full rendered request. Fixed-field subtraction provides a
budget report, not an assumption that separately tokenized fields are additive.
Head/tail retain approximately equal numbers of token boundaries; an odd unit
goes to the head. The binary search need not find a maximal BPE packing: every
accepted candidate is independently measured and never exceeds the budget.
"""
    original = deepcopy(actor(row))
    state = original["state"]
    if not isinstance(state, str):
        raise TypeError("state must be text")
    fixed = {k: v for k, v in original.items() if k != "state"}
    empty = {**original, "state": ""}
    total, fixed_tokens = counter(original), counter(empty)
    meta = dict(protocol=PROTOCOL, budget=asdict(budget), counter=counter.identity,
                available_input_tokens=budget.available, fixed_request_tokens=fixed_tokens,
                nominal_state_token_budget=budget.available-fixed_tokens,
                original_input_tokens=total, original_state_tokens=counter.text_tokens(state),
                original_actor_sha256=digest(original), protected_fields_sha256=digest(fixed),
                original_state_sha256=digest(state), original_state_chars=len(state))
    # Check the actual original first; tokenization is not additive at boundaries.
    if total <= budget.available:
        view = original
        ranges = [[0, len(state)]] if state else []
        marker_used = False
    else:
        if fixed_tokens > budget.available:
            raise FixedFieldsTooLong({**meta, "status": "fixed_fields_exceed_budget"})
        boundaries = counter.boundaries(state)
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != len(state):
            raise ValueError("counter must expose original-text boundaries")
        units = len(boundaries)-1
        view, ranges, marker_used = empty, [], False
        # A marker is counted only if a nonempty head or tail can also fit.
        lo, hi = 1, max(0, units-1)
        while lo <= hi:
            keep = (lo+hi)//2
            head, tail = (keep+1)//2, keep//2
            left, right = boundaries[head], boundaries[units-tail]
            candidate = {**original, "state": state[:left]+MARKER+state[right:]}
            if counter(candidate) <= budget.available:
                view = candidate
                ranges = [[0, left]] + ([[right, len(state)]] if tail else [])
                marker_used = True
                lo = keep+1
            else:
                hi = keep-1
    used = counter(view)
    if used > budget.available:
        raise AssertionError("adapted request exceeds configured budget")
    if {k: v for k, v in view.items() if k != "state"} != fixed:
        raise AssertionError("protected fields changed")
    meta.update(truncated=view["state"] != state, input_tokens=used,
                input_tokens_with_reserves=used+budget.service_overhead+budget.output_reserve,
                state_tokens=counter.text_tokens(view["state"]), state_chars=len(view["state"]),
                kept_original_char_ranges=ranges,
                retained_original_chars=sum(b-a for a, b in ranges),
                marker_used=marker_used, effective_actor_sha256=digest(view),
                effective_state_sha256=digest(view["state"]), status="prepared")
    return view, meta

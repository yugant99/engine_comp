"""Build the four workloads of SPEC section 5 (and the conformance inputs) from a seed and a corpus.

Every prompt is a contiguous corpus excerpt cut to an exact token count: the excerpt is decoded
from a slice of the corpus token ids, stripped, and kept only if re-encoding it gives exactly
the wanted number of tokens. Lengths come from a small set so the reference can batch
equal-length prompts without padding. No two excerpts start at the same corpus position.
"""

import json
import random
from dataclasses import asdict, dataclass, field

WORKLOAD_ORDER = ("single", "c8", "c32", "chat16")
CLIENTS = {"single": 1, "c8": 8, "c32": 32, "chat16": 16}
REQUEST_COUNTS = {
    "full": {"single": 8, "c8": 48, "c32": 160, "chat16": 64},
    "quick": {"single": 2, "c8": 16, "c32": 32, "chat16": 16},
}
OUTPUT_TOKENS = {"single": 128, "c8": 128, "c32": 128, "chat16": 96}
SINGLE_PROMPT_TOKENS = 512
COMPLETION_PROMPT_TOKENS = (64, 128, 256, 384, 512, 768, 1024)
CHAT_EXCERPT_TOKENS = (48, 96, 160)
CHAT_SYSTEM_TOKENS = 300
CHAT_INSTRUCTION = "Summarize the following passage in two sentences.\n\n"

MIN_CORPUS_TOKENS = 20000
CONTEXT_LIMIT = 4096
LONG_OK_PROMPT_TOKENS = 4000  # with 64 output tokens: 4064 <= 4096, must be served
OVER_LENGTH_SLICE_TOKENS = 5200  # far above 4096 under any reading of "longer than 4,096 tokens"
EOS_PROBE_MESSAGE = "Reply with only the word: yes"


class PromptError(Exception):
    pass


@dataclass
class Request:
    id: str
    workload: str
    kind: str  # "completion" or "chat"
    max_tokens: int
    ignore_eos: bool
    prompt: str = None
    messages: list = None
    prompt_tokens: int = 0  # the grader's own count for this prompt (templated for chat)

    @property
    def path(self):
        return "/v1/completions" if self.kind == "completion" else "/v1/chat/completions"

    def body(self, model_name=None):
        """The streaming, greedy request body used for workload requests."""
        b = {}
        if model_name:
            b["model"] = model_name
        if self.kind == "completion":
            b["prompt"] = self.prompt
        else:
            b["messages"] = self.messages
        b.update(max_tokens=self.max_tokens, temperature=0, stream=True,
                 stream_options={"include_usage": True})
        if self.ignore_eos:
            b["ignore_eos"] = True
        return b

    def reference_input(self):
        d = {"id": self.id, "kind": self.kind, "max_tokens": self.max_tokens, "ignore_eos": self.ignore_eos}
        if self.kind == "completion":
            d["prompt"] = self.prompt
        else:
            d["messages"] = self.messages
        return d


@dataclass
class Plan:
    seed: int
    mode: str
    workloads: dict  # name -> list[Request]
    conformance: dict  # name -> inputs (strings / lists of strings)
    reference_extra: list = field(default_factory=list)  # conformance requests that need a reference

    def all_reference_requests(self):
        reqs = [r for name in WORKLOAD_ORDER for r in self.workloads[name]]
        return reqs + list(self.reference_extra)

    def summary(self):
        out = {}
        for name in WORKLOAD_ORDER:
            reqs = self.workloads[name]
            lengths = {}
            for r in reqs:
                lengths[r.prompt_tokens] = lengths.get(r.prompt_tokens, 0) + 1
            out[name] = {"requests": len(reqs), "prompt_token_histogram": dict(sorted(lengths.items()))}
        return out


class ExcerptSampler:
    def __init__(self, corpus_text, tok, rng):
        self.tok = tok
        self.rng = rng
        self.ids = tok.encode(corpus_text)
        if len(self.ids) < MIN_CORPUS_TOKENS:
            raise PromptError("corpus has %d tokens; at least %d are needed" % (len(self.ids), MIN_CORPUS_TOKENS))
        self.used_starts = set()

    def exact(self, n_tokens, max_tries=500):
        """A stripped excerpt whose standalone tokenization is exactly n_tokens long."""
        hi = len(self.ids) - n_tokens - 1
        if hi <= 0:
            raise PromptError("corpus too short for a %d-token excerpt" % n_tokens)
        for _ in range(max_tries):
            start = self.rng.randrange(0, hi)
            if start in self.used_starts:
                continue
            text = self.tok.decode(self.ids[start:start + n_tokens]).strip()
            if not text or "�" in text or "<|" in text:
                continue
            if len(self.tok.encode(text)) != n_tokens:
                continue
            self.used_starts.add(start)
            return text
        raise PromptError("could not cut a clean %d-token excerpt in %d tries" % (n_tokens, max_tries))

    def rough(self, n_tokens):
        """An excerpt of about n_tokens (no exactness needed)."""
        hi = len(self.ids) - n_tokens - 1
        if hi <= 0:
            raise PromptError("corpus too short for a %d-token excerpt" % n_tokens)
        start = self.rng.randrange(0, hi)
        return self.tok.decode(self.ids[start:start + n_tokens]).strip()


def build_plan(seed, corpus_text, tok, mode):
    if mode not in REQUEST_COUNTS:
        raise PromptError("unknown mode %r" % mode)
    rng = random.Random("engine-arena-v1:%d" % seed)
    sampler = ExcerptSampler(corpus_text, tok, rng)
    counts = REQUEST_COUNTS[mode]
    workloads = {}

    workloads["single"] = [
        _completion("single", i, sampler.exact(SINGLE_PROMPT_TOKENS), SINGLE_PROMPT_TOKENS)
        for i in range(counts["single"])
    ]
    for name in ("c8", "c32"):
        reqs = []
        for i in range(counts[name]):
            n = rng.choice(COMPLETION_PROMPT_TOKENS)
            reqs.append(_completion(name, i, sampler.exact(n), n))
        workloads[name] = reqs

    system = sampler.exact(CHAT_SYSTEM_TOKENS)
    chat = []
    for i in range(counts["chat16"]):
        excerpt = sampler.exact(rng.choice(CHAT_EXCERPT_TOKENS))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": CHAT_INSTRUCTION + excerpt}]
        chat.append(Request(id="chat16-%03d" % i, workload="chat16", kind="chat",
                            max_tokens=OUTPUT_TOKENS["chat16"], ignore_eos=False, messages=messages,
                            prompt_tokens=len(tok.chat_prompt_ids(messages))))
    workloads["chat16"] = chat

    over = sampler.rough(OVER_LENGTH_SLICE_TOKENS)
    over_tokens = len(tok.encode(over))
    if over_tokens <= CONTEXT_LIMIT + 500:
        raise PromptError("over-length probe has only %d tokens" % over_tokens)
    conformance = {
        "short": [sampler.exact(64) for _ in range(8)],
        "chat_excerpt": sampler.exact(64),
        "concurrent": [sampler.exact(32) for _ in range(64)],
        "long_ok": sampler.exact(LONG_OK_PROMPT_TOKENS),
        "over_length": over,
        "over_length_tokens": over_tokens,
    }
    eos_probe = Request(id="conf-eos-probe", workload="conformance", kind="chat", max_tokens=32,
                        ignore_eos=False, messages=[{"role": "user", "content": EOS_PROBE_MESSAGE}])
    eos_probe.prompt_tokens = len(tok.chat_prompt_ids(eos_probe.messages))
    return Plan(seed=seed, mode=mode, workloads=workloads, conformance=conformance, reference_extra=[eos_probe])


def _completion(workload, i, text, n_tokens):
    return Request(id="%s-%03d" % (workload, i), workload=workload, kind="completion",
                   max_tokens=OUTPUT_TOKENS[workload], ignore_eos=True, prompt=text, prompt_tokens=n_tokens)


def write_reference_requests(plan, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.reference_input() for r in plan.all_reference_requests()], f)


def request_to_dict(r):
    return asdict(r)

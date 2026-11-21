"""Conformance: one check per behaviour in SPEC section 4.

Checks run one at a time with nothing else in flight (so "same request, same engine state"
holds), except concurrent_64. Every request here is also valid for a vLLM OpenAI server
(it supports ignore_eos, stop, seed and stream_options), because vLLM is graded with the same
code as the baseline. A check marked baseline_exempt is not enforced with --baseline-engine;
currently no check needs that. Checks with enforced=False are informational: they are reported
but never fail a run, because the SPEC gives no threshold for them.
"""

import asyncio
from dataclasses import asdict, dataclass

import aiohttp

from .client import FINISH_REASONS, request, stream_request
from .prompts import CHAT_INSTRUCTION, EOS_PROBE_MESSAGE

CHECK_TIMEOUT_S = 900
NEVER_STOP = "QQXZQQXZ"


class CheckFailed(Exception):
    pass


def expect(cond, message):
    if not cond:
        raise CheckFailed(message)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    spec: str
    enforced: bool = True
    baseline_exempt: bool = False

    def to_dict(self):
        return asdict(self)


def usage_shape(u):
    expect(isinstance(u, dict), "usage missing or not an object: %r" % (u,))
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        expect(isinstance(u.get(k), int), "usage.%s is not an integer: %r" % (k, u))
    expect(u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
           "usage.total_tokens != prompt_tokens + completion_tokens: %r" % u)
    return u


def choose_stop_string(text):
    """A 4-character word start from the later part of text; returns (stop, index of first match)."""
    n = len(text)
    for k in range(n // 3, n - 4):
        if text[k].isspace() or (k > 0 and not text[k - 1].isspace()):
            continue
        s = text[k:k + 4]
        if any(ch.isspace() for ch in s) or "�" in s:
            continue
        idx = text.find(s)
        if idx >= 1:
            return s, idx
    return None, None


class Conformance:
    def __init__(self, base_url, tok, inputs, refs, timeout_s=120.0, model_name=None):
        self.base = base_url.rstrip("/")
        self.tok = tok
        self.inp = inputs
        self.refs = refs
        self.timeout_s = timeout_s
        self.model_name = model_name
        self.ctx = {}
        self.session = None

    # ------------------------------------------------------------------ helpers

    def body(self, **kw):
        b = {"model": self.model_name} if self.model_name else {}
        b["temperature"] = 0
        b.update(kw)
        return b

    async def post(self, path, payload=None, raw=None):
        r = await request(self.session, "POST", self.base + path, payload=payload, raw=raw, timeout_s=self.timeout_s)
        return r

    async def stream(self, path, payload, kind, want_usage=False, keep_times=False):
        return await stream_request(self.session, self.base + path, payload, kind, self.timeout_s,
                                    want_usage=want_usage, keep_times=keep_times)

    def _ok_json(self, r, what):
        expect(r.error is None, "%s: %s" % (what, r.error))
        expect(r.status == 200, "%s: HTTP %s: %s" % (what, r.status, r.body[:300]))
        expect(isinstance(r.json, dict), "%s: response is not a JSON object" % what)
        return r.json

    async def completion(self, what, **kw):
        """POST /v1/completions (non-streaming); validates the response shape."""
        j = self._ok_json(await self.post("/v1/completions", self.body(**kw)), what)
        expect(j.get("object") == "text_completion", "%s: object is %r, not 'text_completion'" % (what, j.get("object")))
        expect(isinstance(j.get("id"), str), "%s: id is not a string" % what)
        expect(isinstance(j.get("model"), str), "%s: model is not a string" % what)
        ch = j.get("choices")
        expect(isinstance(ch, list) and ch and isinstance(ch[0], dict), "%s: choices missing" % what)
        c = ch[0]
        expect(c.get("index") == 0, "%s: choices[0].index is %r" % (what, c.get("index")))
        expect(isinstance(c.get("text"), str), "%s: choices[0].text is not a string" % what)
        expect(c.get("finish_reason") in FINISH_REASONS, "%s: finish_reason %r" % (what, c.get("finish_reason")))
        return c["text"], c["finish_reason"], usage_shape(j.get("usage"))

    async def chat(self, what, **kw):
        j = self._ok_json(await self.post("/v1/chat/completions", self.body(**kw)), what)
        ch = j.get("choices")
        expect(isinstance(ch, list) and ch and isinstance(ch[0], dict), "%s: choices missing" % what)
        c = ch[0]
        m = c.get("message")
        expect(isinstance(m, dict), "%s: choices[0].message missing" % what)
        expect(m.get("role") == "assistant", "%s: message.role is %r" % (what, m.get("role")))
        expect(isinstance(m.get("content"), str), "%s: message.content is not a string" % what)
        expect(c.get("finish_reason") in FINISH_REASONS, "%s: finish_reason %r" % (what, c.get("finish_reason")))
        return m["content"], c["finish_reason"], usage_shape(j.get("usage"))

    async def still_serving(self):
        h = await request(self.session, "GET", self.base + "/health", timeout_s=self.timeout_s)
        if h.status != 200:
            return "health returned %s %s" % (h.status, h.error or "")
        r = await self.post("/v1/completions", self.body(prompt="Hello", max_tokens=1))
        if r.status != 200:
            return "1-token completion returned %s %s" % (r.status, r.error or "")
        return None

    # ------------------------------------------------------------------ checks

    async def c_health(self):
        r = await request(self.session, "GET", self.base + "/health", timeout_s=self.timeout_s)
        expect(r.status == 200, "GET /health: %s %s" % (r.status, r.error or ""))
        return "200"

    async def c_completions_nonstream(self):
        p = self.inp["short"][0]
        text, finish, u = await self.completion("completion", prompt=p, max_tokens=16)
        n = len(self.tok.encode(p))
        expect(u["prompt_tokens"] == n, "usage.prompt_tokens %d, grader counts %d" % (u["prompt_tokens"], n))
        expect(u["completion_tokens"] <= 16, "completion_tokens %d > max_tokens 16" % u["completion_tokens"])
        if finish == "length":
            expect(u["completion_tokens"] == 16, "finish_reason length but completion_tokens %d" % u["completion_tokens"])
        self.ctx["completion"] = (text, finish)
        return "ok (finish_reason %s, %d tokens)" % (finish, u["completion_tokens"])

    async def c_completions_stream(self):
        expect("completion" in self.ctx, "needs a valid non-streaming response (completions_nonstream)")
        rec = await self.stream("/v1/completions", self.body(prompt=self.inp["short"][0], max_tokens=16, stream=True),
                                "completion")
        expect(rec.error is None, rec.error or "")
        expect(rec.events_after_done == 0, "%d event(s) after [DONE]" % rec.events_after_done)
        text, finish = self.ctx["completion"]
        expect(rec.text == text, "streamed text differs from the non-streaming text: %r vs %r" % (rec.text[:120], text[:120]))
        expect(rec.finish_reason == finish, "streamed finish_reason %r vs %r" % (rec.finish_reason, finish))
        return "ok (%d events)" % rec.n_events

    async def c_include_usage(self):
        p = self.inp["short"][1]
        rec = await self.stream("/v1/completions", self.body(prompt=p, max_tokens=24, ignore_eos=True, stream=True,
                                                             stream_options={"include_usage": True}),
                                "completion", want_usage=True)
        expect(rec.error is None, rec.error or "")
        expect(rec.usage_last, "usage is not on the last event before [DONE]")
        u = usage_shape(rec.usage)
        n = len(self.tok.encode(p))
        expect(u["prompt_tokens"] == n, "usage.prompt_tokens %d, grader counts %d" % (u["prompt_tokens"], n))
        expect(u["completion_tokens"] == 24, "completion_tokens %d with ignore_eos and max_tokens 24" % u["completion_tokens"])
        expect(rec.finish_reason == "length", "finish_reason %r" % rec.finish_reason)
        return "ok"

    async def c_max_tokens(self):
        text, finish, u = await self.completion("completion", prompt=self.inp["short"][2], max_tokens=8)
        expect(u["completion_tokens"] <= 8, "completion_tokens %d > max_tokens 8" % u["completion_tokens"])
        if finish == "length":
            expect(u["completion_tokens"] == 8, "finish_reason length but completion_tokens %d" % u["completion_tokens"])
        n = len(self.tok.encode(text))
        expect(n <= 16, "returned text re-tokenizes to %d tokens for max_tokens 8" % n)
        return "ok (%d tokens, text re-tokenizes to %d)" % (u["completion_tokens"], n)

    async def c_ignore_eos(self):
        _, finish, u = await self.completion("completion", prompt=self.inp["short"][3], max_tokens=48, ignore_eos=True)
        expect(u["completion_tokens"] == 48 and finish == "length",
               "completions: %d tokens, finish_reason %r; want 48, 'length'" % (u["completion_tokens"], finish))
        msgs = [{"role": "user", "content": EOS_PROBE_MESSAGE}]
        _, finish, u = await self.chat("chat", messages=msgs, max_tokens=24, ignore_eos=True)
        expect(u["completion_tokens"] == 24 and finish == "length",
               "chat: %d tokens, finish_reason %r; want 24, 'length'" % (u["completion_tokens"], finish))
        return "ok"

    async def c_eos_stop(self):
        ref = self.refs.get("conf-eos-probe")
        msgs = [{"role": "user", "content": EOS_PROBE_MESSAGE}]
        text, finish, u = await self.chat("chat", messages=msgs, max_tokens=32)
        if not ref or not ref["eos_stopped"]:
            return "reference did not stop on EOS within 32 tokens; finish_reason not enforced (got %r)" % finish
        expect(finish == "stop", "reference stops on EOS after %d tokens (%r) but finish_reason is %r, %d tokens"
               % (ref["generated_tokens"], ref["text"][:60], finish, u["completion_tokens"]))
        return "ok (%r)" % text[:40]

    async def c_stop_strings(self):
        # ignore_eos keeps the greedy output long enough to pick a stop string from; it only
        # disables EOS, never stop strings.
        for p in self.inp["short"][4:7]:
            full, _, _ = await self.completion("completion", prompt=p, max_tokens=48, ignore_eos=True)
            stop, idx = choose_stop_string(full)
            if stop:
                break
        else:
            raise CheckFailed("could not choose a stop string from three greedy outputs")
        want = full[:idx]
        kw = dict(prompt=p, max_tokens=48, ignore_eos=True, stop=[stop, NEVER_STOP])
        text, finish, _ = await self.completion("completion with stop", **kw)
        expect(text == want, "stop %r: got %r, want %r" % (stop, text[-80:], want[-80:]))
        expect(finish == "stop", "stop %r matched but finish_reason is %r" % (stop, finish))
        rec = await self.stream("/v1/completions", self.body(stream=True, **kw), "completion")
        expect(rec.error is None, "streaming: %s" % rec.error)
        expect(rec.text == want, "streaming, stop %r: got %r, want %r" % (stop, rec.text[-80:], want[-80:]))
        expect(rec.finish_reason == "stop", "streaming: finish_reason %r" % rec.finish_reason)
        return "ok (stop %r after %d chars)" % (stop, idx)

    async def c_greedy_determinism(self):
        p = self.inp["short"][7]
        a, _, _ = await self.completion("first", prompt=p, max_tokens=32)
        b, _, _ = await self.completion("second", prompt=p, max_tokens=32)
        expect(a == b, "two identical temperature-0 requests differ: %r vs %r" % (a[:120], b[:120]))
        self.ctx["greedy32"] = a
        return "ok"

    async def c_seeded_sampling(self):
        # ignore_eos so that an early EOS cannot make the sample empty.
        kw = dict(prompt=self.inp["short"][7], max_tokens=32, temperature=0.8, top_p=0.95, seed=1234, ignore_eos=True)
        a, _, _ = await self.completion("first", **kw)
        b, _, _ = await self.completion("second", **kw)
        expect(a.strip() != "", "sampled text is empty")
        expect(a == b, "same seed gave different samples: %r vs %r" % (a[:120], b[:120]))
        return "ok"

    async def c_top_p(self):
        expect("greedy32" in self.ctx, "needs greedy_determinism")
        a, _, _ = await self.completion("completion", prompt=self.inp["short"][7], max_tokens=32,
                                        temperature=1.0, top_p=1e-6, seed=7)
        expect(a == self.ctx["greedy32"], "top_p 1e-6 should keep only the top token, but the text differs from "
               "greedy: %r vs %r" % (a[:120], self.ctx["greedy32"][:120]))
        return "ok"

    async def c_chat_nonstream(self):
        msgs = [{"role": "user", "content": CHAT_INSTRUCTION + self.inp["chat_excerpt"]}]
        text, finish, u = await self.chat("chat", messages=msgs, max_tokens=32)
        n = len(self.tok.chat_prompt_ids(msgs))
        expect(u["prompt_tokens"] == n, "usage.prompt_tokens %d, templated prompt has %d tokens" % (u["prompt_tokens"], n))
        expect(u["completion_tokens"] <= 32, "completion_tokens %d > max_tokens 32" % u["completion_tokens"])
        self.ctx["chat"] = (msgs, text, finish)
        return "ok (%d prompt tokens)" % n

    async def c_chat_stream(self):
        expect("chat" in self.ctx, "needs a valid non-streaming chat response (chat_nonstream)")
        msgs, text, finish = self.ctx["chat"]
        rec = await self.stream("/v1/chat/completions",
                                self.body(messages=msgs, max_tokens=32, stream=True, stream_options={"include_usage": True}),
                                "chat", want_usage=True)
        expect(rec.error is None, rec.error or "")
        expect(all(r == "assistant" for r in rec.roles), "delta.role values %r" % rec.roles)
        expect(rec.events_after_done == 0, "%d event(s) after [DONE]" % rec.events_after_done)
        expect(rec.text == text, "streamed content differs from non-streaming: %r vs %r" % (rec.text[:120], text[:120]))
        expect(rec.finish_reason == finish, "streamed finish_reason %r vs %r" % (rec.finish_reason, finish))
        expect(rec.usage_last, "usage is not on the last event before [DONE]")
        n = len(self.tok.chat_prompt_ids(msgs))
        expect(usage_shape(rec.usage)["prompt_tokens"] == n,
               "usage.prompt_tokens %d, templated prompt has %d tokens" % (rec.usage["prompt_tokens"], n))
        return "ok"

    async def c_streaming_granularity(self):
        rec = await self.stream("/v1/completions",
                                self.body(prompt=self.inp["short"][0], max_tokens=64, ignore_eos=True, stream=True),
                                "completion", keep_times=True)
        expect(rec.error is None, rec.error or "")
        credited = min(len(self.tok.encode(rec.text)), 64)
        if rec.t_first_text is not None and rec.t_end > rec.t_send:
            self.ctx["spread"] = (rec.t_last_text - rec.t_first_text) / (rec.t_end - rec.t_send)
        expect(rec.n_text_events >= 0.5 * credited,
               "%d text events for %d tokens: output is sent in bursts" % (rec.n_text_events, credited))
        return "ok (%d text events, %d tokens)" % (rec.n_text_events, credited)

    async def c_streaming_spread(self):
        expect("spread" in self.ctx, "needs streaming_granularity")
        s = self.ctx["spread"]
        expect(s >= 0.5, "text events span only %.0f%% of the request time" % (100 * s))
        return "text events span %.0f%% of the request time" % (100 * s)

    async def c_long_prompt(self):
        p = self.inp["long_ok"]
        n = len(self.tok.encode(p))
        _, finish, u = await self.completion("completion", prompt=p, max_tokens=64, ignore_eos=True)
        expect(u["prompt_tokens"] == n, "usage.prompt_tokens %d, grader counts %d" % (u["prompt_tokens"], n))
        expect(u["completion_tokens"] == 64 and finish == "length",
               "%d tokens, finish_reason %r; want 64, 'length'" % (u["completion_tokens"], finish))
        return "ok (%d + 64 tokens)" % n

    async def c_errors(self):
        n_over = self.inp["over_length_tokens"]
        cases = [
            ("malformed JSON on /v1/completions", "/v1/completions", None, b'{"prompt": "Hello", "max_tokens": '),
            ("malformed JSON on /v1/chat/completions", "/v1/chat/completions", None,
             b'{"messages": [{"role": "user", "content": "Hi"}'),
            ("missing prompt", "/v1/completions", self.body(max_tokens=8), None),
            ("missing messages", "/v1/chat/completions", self.body(max_tokens=8), None),
            ("prompt of %d tokens" % n_over, "/v1/completions",
             self.body(prompt=self.inp["over_length"], max_tokens=16), None),
        ]
        problems = []
        for what, path, payload, raw in cases:
            r = await self.post(path, payload=payload, raw=raw)
            if r.error:
                problems.append("%s: %s" % (what, r.error))
            elif r.status != 400:
                problems.append("%s: HTTP %s, want 400" % (what, r.status))
            elif r.json is None:
                problems.append("%s: 400 without a JSON body" % what)
            after = await self.still_serving()
            if after:
                problems.append("after %s: %s" % (what, after))
        expect(not problems, "; ".join(problems))
        return "ok (%d cases)" % len(cases)

    async def c_concurrent_64(self):
        prompts = self.inp["concurrent"]
        connector = aiohttp.TCPConnector(limit=len(prompts) + 4)
        async with aiohttp.ClientSession(connector=connector) as s:
            recs = await asyncio.gather(*[
                stream_request(s, self.base + "/v1/completions",
                               self.body(prompt=p, max_tokens=16, stream=True, stream_options={"include_usage": True}),
                               "completion", self.timeout_s, want_usage=True)
                for p in prompts])
        bad = [r.error for r in recs if r.error]
        expect(not bad, "%d of %d failed, first: %s" % (len(bad), len(recs), bad[0] if bad else ""))
        return "ok (%d requests)" % len(recs)

    # ------------------------------------------------------------------ driver

    CHECKS = [
        ("health", "c_health", "GET /health: 200 once the model is loaded", True),
        ("completions_nonstream", "c_completions_nonstream", "POST /v1/completions non-streaming response shape and usage", True),
        ("completions_stream", "c_completions_stream", "streaming: data: {json} events, choices[0].text, then data: [DONE]; last event with choices carries finish_reason", True),
        ("stream_include_usage", "c_include_usage", "stream_options.include_usage: the last event before [DONE] carries usage", True),
        ("max_tokens", "c_max_tokens", "max_tokens: upper bound on generated tokens", True),
        ("ignore_eos", "c_ignore_eos", "ignore_eos: never stop on EOS; generate exactly max_tokens tokens", True),
        ("eos_stop", "c_eos_stop", "finish_reason is 'stop' when the model emitted an end-of-sequence token", True),
        ("stop_strings", "c_stop_strings", "stop: stop before emitting any of these; the stop string is not included", True),
        ("greedy_determinism", "c_greedy_determinism", "temperature 0 means greedy (argmax)", True),
        ("seeded_sampling", "c_seeded_sampling", "seed: same seed, same request, same engine state gives the same sample", True),
        ("top_p", "c_top_p", "top_p: nucleus sampling (tiny top_p keeps only the top token); informational", False),
        ("chat_nonstream", "c_chat_nonstream", "chat: choices[0].message; usage.prompt_tokens equals the templated prompt", True),
        ("chat_stream", "c_chat_stream", "chat streaming: choices[0].delta.content; first event may carry delta.role", True),
        ("streaming_granularity", "c_streaming_granularity", "each token flushed as soon as decodable; no bursts (text events >= 0.5 per token)", True),
        ("streaming_spread", "c_streaming_spread", "no holding back output: text events spread over the request; informational", False),
        ("long_prompt_accepted", "c_long_prompt", "requests up to 4,096 tokens (prompt + output) are served", True),
        ("errors", "c_errors", "malformed JSON, missing prompt, over-length: HTTP 400 with a JSON body; server keeps serving", True),
        ("concurrent_64", "c_concurrent_64", "64 concurrent short requests all succeed", True),
    ]
    BASELINE_EXEMPT = set()

    async def run(self):
        results = []
        connector = aiohttp.TCPConnector(limit=8)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            for name, method, spec, enforced in self.CHECKS:
                try:
                    detail = await asyncio.wait_for(getattr(self, method)(), CHECK_TIMEOUT_S)
                    passed = True
                except CheckFailed as e:
                    passed, detail = False, str(e)
                except asyncio.TimeoutError:
                    passed, detail = False, "check did not finish within %d s" % CHECK_TIMEOUT_S
                except Exception as e:  # a bug in a check must show up, not crash the run
                    passed, detail = False, "grader exception: %s: %s" % (type(e).__name__, e)
                results.append(CheckResult(name=name, passed=passed, detail=str(detail)[:1000], spec=spec,
                                           enforced=enforced, baseline_exempt=name in self.BASELINE_EXEMPT))
        return results


def failing(results, baseline_engine=False):
    """Enforced checks that failed (baseline_exempt ones are skipped for the baseline engine)."""
    return [r for r in results if not r.passed and r.enforced and not (baseline_engine and r.baseline_exempt)]

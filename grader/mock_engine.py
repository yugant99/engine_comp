"""A small engine implementing the SPEC API on top of grader.fakemodel, for grader tests.

    python -m grader.mock_engine --host 127.0.0.1 --port 8000 [--fault NAME[:ARG]] ...

Behaviour matches what the grader expects of a correct engine: whitespace tokenization, text
stop strings with hold-back while streaming, EOS counted in usage, HTTP 400 with a JSON body for
bad requests. Faults (comma-separated in --fault) make it misbehave so tests can show the
grader catches each one. Faults that touch generation apply only to requests with
max_tokens >= --fault-min-tokens (default 0, i.e. all), which lets a fault slip past
conformance (short requests) and hit only the workloads (128 / 96 tokens).

    wrong-after:N   text is wrong after N generated tokens
    mixup           a request is answered with the previous request's prompt
    bursty          text is sent 8 tokens per event
    hold-all        all events are sent at the end of the request
    bad-usage       usage.completion_tokens is too high
    error500:K      every K-th request answers HTTP 500
    hang:N          stop sending after N tokens and never finish
    eos-ignore      never stop on EOS (as if ignore_eos were always set)
    stop-include    the matched stop string is included in the output
    no-health       /health never returns 200
    ignore-sigterm  SIGTERM is ignored
"""

import argparse
import asyncio
import json
import os
import signal
import time
import uuid

from aiohttp import web

from . import fakemodel

CONTEXT_LIMIT = 4096


class BadRequest(Exception):
    pass


class State:
    def __init__(self, args):
        self.args = args
        self.faults = {}
        for item in (args.fault or "").split(","):
            item = item.strip()
            if item:
                name, _, arg = item.partition(":")
                self.faults[name] = int(arg) if arg else None
        self.t_start = time.monotonic()
        self.last_context = None
        self.eligible_count = 0

    def fault(self, name, max_tokens):
        if name not in self.faults:
            return False
        return max_tokens >= self.args.fault_min_tokens


def error_response(status, message):
    return web.json_response({"object": "error", "message": message, "type": "BadRequestError", "code": status},
                             status=status)


def parse_request(raw, kind):
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise BadRequest("malformed JSON: %s" % e)
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    if kind == "completion":
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise BadRequest("prompt (string) is required")
        context = prompt.split()
    else:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise BadRequest("messages (list of objects) is required")
        context = fakemodel.chat_template_words(messages)
    max_tokens = body.get("max_tokens", 16)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("max_tokens must be a positive integer")
    if len(context) + max_tokens > CONTEXT_LIMIT:
        raise BadRequest("request has %d prompt tokens + %d max_tokens > %d" % (len(context), max_tokens, CONTEXT_LIMIT))
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    return {
        "context": context,
        "max_tokens": max_tokens,
        "temperature": float(body.get("temperature", 1.0)),
        "top_p": float(body.get("top_p", 1.0)),
        "seed": body.get("seed"),
        "stop": [s for s in stop if s],
        "ignore_eos": bool(body.get("ignore_eos", False)),
        "stream": bool(body.get("stream", False)),
        "include_usage": bool((body.get("stream_options") or {}).get("include_usage", False)),
    }


async def generate(state, p):
    """Yields text pieces; returns via the final dict in p: tokens, finish_reason."""
    args = state.args
    mt = p["max_tokens"]
    context = p["context"]
    if state.fault("mixup", mt) and state.last_context is not None:
        context = state.last_context
    state.last_context = p["context"]
    gen = fakemodel.Generation(context, p["temperature"], p["top_p"], p["seed"])
    ignore_eos = p["ignore_eos"] or state.fault("eos-ignore", mt)
    wrong_after = state.faults.get("wrong-after") if state.fault("wrong-after", mt) else None
    hang_after = state.faults.get("hang") if state.fault("hang", mt) else None
    stops = p["stop"]
    hold = max((len(s) for s in stops), default=1) - 1
    text, sent = "", 0
    p["tokens"], p["finish_reason"] = 0, "length"
    for _ in range(mt):
        if args.token_delay:
            await asyncio.sleep(args.token_delay)
        if hang_after is not None and p["tokens"] >= hang_after:
            await asyncio.sleep(3600)
        tok = gen.next_token()
        p["tokens"] += 1
        if tok == fakemodel.EOS and not ignore_eos:
            p["finish_reason"] = "stop"
            break
        gen.advance(tok)
        piece = fakemodel.token_text(tok)
        if wrong_after is not None and p["tokens"] > wrong_after and piece:
            i = fakemodel.VOCAB.index(tok) if tok in fakemodel.VOCAB else 0
            piece = " " + fakemodel.VOCAB[(i + 1) % len(fakemodel.VOCAB)]
        text += piece
        hits = [(text.find(s), s) for s in stops if s in text]
        if hits:
            idx, s = min(hits)
            text = text[:idx + len(s)] if state.fault("stop-include", mt) else text[:idx]
            p["finish_reason"] = "stop"
            break
        if len(text) - hold > sent:
            yield text[sent:len(text) - hold]
            sent = len(text) - hold
    if len(text) > sent:
        yield text[sent:]


def usage_of(state, p):
    n = p["tokens"]
    if state.fault("bad-usage", p["max_tokens"]):
        n += max(3, n // 10)
    prompt = len(p["context"])
    return {"prompt_tokens": prompt, "completion_tokens": n, "total_tokens": prompt + n}


def make_handler(state, kind):
    async def handler(request):
        try:
            p = parse_request(await request.read(), kind)
        except BadRequest as e:
            return error_response(400, str(e))
        if state.fault("error500", p["max_tokens"]):
            state.eligible_count += 1
            if state.eligible_count % (state.faults["error500"] or 1) == 0:
                return error_response(500, "injected internal error")
        rid = ("cmpl-" if kind == "completion" else "chatcmpl-") + uuid.uuid4().hex[:16]
        created = int(time.time())
        if not p["stream"]:
            text = "".join([piece async for piece in generate(state, p)])
            if kind == "completion":
                choice = {"index": 0, "text": text, "finish_reason": p["finish_reason"]}
                obj = "text_completion"
            else:
                choice = {"index": 0, "message": {"role": "assistant", "content": text},
                          "finish_reason": p["finish_reason"]}
                obj = "chat.completion"
            return web.json_response({"id": rid, "object": obj, "created": created, "model": "mock",
                                      "choices": [choice], "usage": usage_of(state, p)})
        return await stream_response(state, request, kind, p, rid, created)
    return handler


async def stream_response(state, request, kind, p, rid, created):
    args = state.args
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    obj = "text_completion" if kind == "completion" else "chat.completion.chunk"
    mt = p["max_tokens"]
    bursty = state.fault("bursty", mt)
    hold_all = state.fault("hold-all", mt)

    def event(choice=None, usage=None):
        d = {"id": rid, "object": obj, "created": created, "model": "mock",
             "choices": [] if choice is None else [choice]}
        if usage is not None:
            d["usage"] = usage
        return ("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode("utf-8")

    def piece_choice(piece, finish=None):
        if kind == "completion":
            return {"index": 0, "text": piece, "finish_reason": finish}
        return {"index": 0, "delta": {"content": piece} if piece else {}, "finish_reason": finish}

    async def send(data):
        if args.chunk_bytes:
            for i in range(0, len(data), args.chunk_bytes):
                await resp.write(data[i:i + args.chunk_bytes])
                await asyncio.sleep(0.0005)
        else:
            await resp.write(data)

    if kind == "chat":
        await send(event({"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}))
    buffered = []
    async for piece in generate(state, p):
        if bursty or hold_all:
            buffered.append(piece)
            if bursty and len(buffered) >= 8:
                await send(event(piece_choice("".join(buffered))))
                buffered = []
        else:
            await send(event(piece_choice(piece)))
    if hold_all:
        for piece in buffered:
            await send(event(piece_choice(piece)))
    elif buffered:
        await send(event(piece_choice("".join(buffered))))
    await send(event(piece_choice("", p["finish_reason"])))
    if p["include_usage"]:
        await send(event(None, usage_of(state, p)))
    await send(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def build_app(state):
    app = web.Application(client_max_size=64 * 1024 * 1024)

    async def health(request):
        if "no-health" in state.faults or time.monotonic() - state.t_start < state.args.load_delay:
            return web.json_response({"status": "loading"}, status=503)
        return web.json_response({"status": "ok"})

    app.router.add_get("/health", health)
    app.router.add_post("/v1/completions", make_handler(state, "completion"))
    app.router.add_post("/v1/chat/completions", make_handler(state, "chat"))
    return app


async def serve(args):
    state = State(args)
    runner = web.AppRunner(build_app(state), handle_signals=False, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print("mock engine listening on %s:%d faults=%s" % (args.host, args.port, state.faults), flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    if "ignore-sigterm" in state.faults:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    await stop.wait()
    try:
        await asyncio.wait_for(runner.cleanup(), 2)
    except asyncio.TimeoutError:
        pass
    print("mock engine stopped", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mock engine for grader tests.")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--token-delay", type=float, default=0.002, help="seconds per generated token")
    ap.add_argument("--load-delay", type=float, default=0.3, help="seconds before /health returns 200")
    ap.add_argument("--chunk-bytes", type=int, default=0, help="write SSE output in pieces of this many bytes")
    ap.add_argument("--fault", default="", help="comma-separated faults, see the module docstring")
    ap.add_argument("--fault-min-tokens", type=int, default=0)
    args = ap.parse_args(argv)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()

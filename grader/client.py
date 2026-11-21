"""HTTP client: streaming (SSE) and plain requests, and the closed-loop workload runner.

All timestamps are time.perf_counter() taken on the one event loop that runs the requests. A
chunk's arrival time is taken as soon as aiohttp hands it over, and every event completed by
that chunk gets that time. Only the cheap parsing needed to timestamp events (JSON decode, text
extraction) happens while a workload runs; correctness checks happen after it finishes.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp

JSON_HEADERS = {"Content-Type": "application/json"}
MAX_STREAM_BYTES = 32 * 1024 * 1024
FINISH_REASONS = ("stop", "length")


class SSEParser:
    """Incremental text/event-stream parser working on raw bytes.

    Lines are split on b"\\n" (a trailing b"\\r" is dropped) and decoded as UTF-8 only once
    complete, so a multi-byte character split across TCP packets is handled. A blank line
    dispatches the event; its data is the event's "data:" lines joined by "\\n". Comment lines
    and other fields are ignored. An event not terminated by a blank line is never dispatched
    (as in the SSE standard); pending() reports one.
    """

    def __init__(self):
        self._buf = bytearray()
        self._data = []

    def feed(self, chunk):
        self._buf += chunk
        events = []
        pos = 0
        while True:
            nl = self._buf.find(b"\n", pos)
            if nl < 0:
                break
            line = bytes(self._buf[pos:nl])
            pos = nl + 1
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data = []
                continue
            if line.startswith(b":"):
                continue
            name, _, value = line.partition(b":")
            if value.startswith(b" "):
                value = value[1:]
            if name == b"data":
                self._data.append(value.decode("utf-8"))  # UnicodeDecodeError: caller reports it
        if pos:
            del self._buf[:pos]
        return events

    def pending(self):
        return bool(self._data) or bool(bytes(self._buf).strip())


@dataclass
class Record:
    """Everything the grader keeps about one streamed request."""

    id: str = ""
    kind: str = "completion"
    t_send: float = 0.0
    t_first_text: float = None
    t_last_text: float = None
    t_done: float = None
    t_end: float = None
    status: int = None
    error: str = None
    n_events: int = 0  # JSON events
    n_choice_events: int = 0
    n_text_events: int = 0  # events with non-empty text
    finish_reason: str = None  # finish_reason of the last event that has a choices entry
    usage: dict = None
    usage_last: bool = False  # usage was on the last JSON event before [DONE]
    roles: list = field(default_factory=list)
    done: bool = False
    events_after_done: int = 0
    text_times: list = None
    _parts: list = field(default_factory=list)

    @property
    def text(self):
        return "".join(self._parts)

    def summary(self, t0=0.0):
        def rel(t):
            return None if t is None else round(t - t0, 6)
        return {
            "id": self.id, "status": self.status, "error": self.error,
            "t_send": rel(self.t_send), "t_first_text": rel(self.t_first_text),
            "t_last_text": rel(self.t_last_text), "t_end": rel(self.t_end),
            "n_events": self.n_events, "n_text_events": self.n_text_events,
            "finish_reason": self.finish_reason, "usage": self.usage, "text": self.text,
        }


def _handle_event(rec, data, now):
    if rec.done:
        rec.events_after_done += 1
        return
    if data == "[DONE]":
        rec.done = True
        rec.t_done = now
        return
    try:
        obj = json.loads(data)
    except ValueError:
        rec.error = "event is not JSON: %r" % data[:200]
        return
    if not isinstance(obj, dict):
        rec.error = "event is not a JSON object: %r" % data[:200]
        return
    if obj.get("error"):
        rec.error = "error event: %s" % json.dumps(obj["error"])[:300]
        return
    rec.n_events += 1
    usage = obj.get("usage")
    rec.usage_last = usage is not None
    if usage is not None:
        rec.usage = usage
    choices = obj.get("choices")
    if choices is None:
        if usage is None:
            rec.error = "event has neither choices nor usage: %r" % data[:200]
        return
    if not isinstance(choices, list):
        rec.error = "choices is not a list: %r" % data[:200]
        return
    if not choices:
        return  # e.g. the usage-only event before [DONE]
    c = choices[0]
    if not isinstance(c, dict):
        rec.error = "choices[0] is not an object: %r" % data[:200]
        return
    rec.n_choice_events += 1
    if rec.kind == "completion":
        piece = c.get("text")
    else:
        delta = c.get("delta")
        if delta is None:
            delta = {}
        if not isinstance(delta, dict):
            rec.error = "choices[0].delta is not an object: %r" % data[:200]
            return
        if delta.get("role") is not None:
            rec.roles.append(delta.get("role"))
        piece = delta.get("content")
    if piece is not None and not isinstance(piece, str):
        rec.error = "text is not a string: %r" % data[:200]
        return
    if piece:
        rec._parts.append(piece)
        rec.n_text_events += 1
        if rec.t_first_text is None:
            rec.t_first_text = now
        rec.t_last_text = now
        if rec.text_times is not None:
            rec.text_times.append(now)
    rec.finish_reason = c.get("finish_reason")


def validate_stream(rec, want_usage):
    """Protocol checks every streamed request must pass; sets rec.error on the first problem."""
    if rec.error:
        return
    if not rec.done:
        rec.error = "stream ended without data: [DONE]"
    elif rec.n_choice_events == 0:
        rec.error = "stream had no event with a choices entry"
    elif rec.finish_reason not in FINISH_REASONS:
        rec.error = "last event with a choices entry has finish_reason %r" % (rec.finish_reason,)
    elif want_usage:
        u = rec.usage
        if not isinstance(u, dict):
            rec.error = "stream_options.include_usage was set but no usage was sent"
        elif not all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens")):
            rec.error = "usage lacks integer prompt_tokens/completion_tokens: %r" % (u,)


async def stream_request(session, url, payload, kind, timeout_s, rid="", want_usage=True, keep_times=False):
    """POST a streaming request and record it. Never raises for request-level failures."""
    if not isinstance(payload, (bytes, bytearray)):
        payload = json.dumps(payload).encode("utf-8")
    rec = Record(id=rid, kind=kind)
    if keep_times:
        rec.text_times = []
    parser = SSEParser()
    rec.t_send = time.perf_counter()
    try:
        async with session.post(url, data=payload, headers=JSON_HEADERS,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            rec.status = resp.status
            if resp.status != 200:
                body = await resp.content.read(4096)
                rec.error = "HTTP %d: %s" % (resp.status, body.decode("utf-8", "replace")[:300])
            else:
                received = 0
                async for chunk in resp.content.iter_any():
                    now = time.perf_counter()
                    received += len(chunk)
                    if received > MAX_STREAM_BYTES:
                        rec.error = "response larger than %d bytes" % MAX_STREAM_BYTES
                        break
                    for data in parser.feed(chunk):
                        _handle_event(rec, data, now)
                        if rec.error:
                            break
                    if rec.error:
                        break
                if not rec.error and parser.pending():
                    rec.error = "stream ended in the middle of an event"
    except asyncio.TimeoutError:
        if rec.done:
            rec.error = "timeout after %g s: the response did not end after data: [DONE]" % timeout_s
        else:
            rec.error = "timeout after %g s" % timeout_s
    except UnicodeDecodeError as e:
        rec.error = "invalid UTF-8 in event stream: %s" % e
    except aiohttp.ClientError as e:
        rec.error = "connection error: %s: %s" % (type(e).__name__, e)
    except OSError as e:
        rec.error = "connection error: %s: %s" % (type(e).__name__, e)
    rec.t_end = time.perf_counter()
    validate_stream(rec, want_usage)
    return rec


@dataclass
class Response:
    status: int = None
    body: bytes = b""
    json: object = None
    error: str = None
    elapsed_s: float = 0.0

    @property
    def is_json(self):
        return self.json is not None


async def request(session, method, url, payload=None, timeout_s=120.0, raw=None):
    """A non-streaming request. payload is JSON-encoded; raw bytes are sent as is."""
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode("utf-8"))
    r = Response()
    t0 = time.perf_counter()
    try:
        async with session.request(method, url, data=data, headers=JSON_HEADERS if data is not None else None,
                                   timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            r.status = resp.status
            r.body = await resp.read()
    except asyncio.TimeoutError:
        r.error = "timeout after %g s" % timeout_s
    except (aiohttp.ClientError, OSError) as e:
        r.error = "connection error: %s: %s" % (type(e).__name__, e)
    r.elapsed_s = time.perf_counter() - t0
    if r.body:
        try:
            r.json = json.loads(r.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            r.json = None
    return r


class RequestFailed(Exception):
    def __init__(self, request, record):
        super().__init__("%s: %s" % (request.id, record.error))
        self.request = request
        self.record = record


@dataclass
class LoopResult:
    records: dict  # request id -> Record, completed requests only
    t_first_send: float = None
    t_last_end: float = None
    failure: str = None
    failed_request: str = None


async def run_closed_loop(base_url, requests, clients, timeout_s, model_name=None, stage_timeout_s=None):
    """N clients pull requests from one shared queue; each sends its next as soon as one finishes.

    Stops at the first failed request: in-flight requests are cancelled and the failure is
    returned. Request bodies are serialised before the clock starts.
    """
    queue = deque((r, base_url + r.path, json.dumps(r.body(model_name)).encode("utf-8")) for r in requests)
    records = {}
    n_clients = max(1, min(clients, len(requests)))
    if stage_timeout_s is None:
        stage_timeout_s = (len(requests) // n_clients + 2) * timeout_s + 60
    connector = aiohttp.TCPConnector(limit=n_clients, limit_per_host=n_clients)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def client():
            while queue:
                req, url, payload = queue.popleft()
                rec = await stream_request(session, url, payload, req.kind, timeout_s, rid=req.id, want_usage=True)
                records[req.id] = rec
                if rec.error:
                    raise RequestFailed(req, rec)

        tasks = [asyncio.ensure_future(client()) for _ in range(n_clients)]
        done, pending = await asyncio.wait(tasks, timeout=stage_timeout_s, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    result = LoopResult(records=records)
    for t in done:
        exc = t.exception()
        if exc is not None:
            if isinstance(exc, RequestFailed):
                result.failure = str(exc)
                result.failed_request = exc.request.id
            else:
                result.failure = "client error: %s: %s" % (type(exc).__name__, exc)
            break
    if result.failure is None and pending:
        result.failure = "workload did not finish within %g s" % stage_timeout_s
    if records:
        result.t_first_send = min(r.t_send for r in records.values())
        result.t_last_end = max(r.t_end for r in records.values())
    return result

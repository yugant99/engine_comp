"""Reference outputs: greedy decoding of the same model by a plain PyTorch implementation.

Run as its own process so all GPU memory is released before the engine starts:

    python -m grader.reference --provider hf --model-dir DIR --requests in.json --out out.json

Input: a JSON list of {"id", "kind": "completion"|"chat", "prompt" | "messages", "max_tokens",
"ignore_eos"}. Output: {"meta": {...}, "results": {id: {...}}} where each result has

    prompt_tokens       prompt length in tokens (chat: the templated prompt)
    generated_ids       generated token ids, without the terminating EOS
    generated_tokens    tokens generated, counting the terminating EOS when there is one
    eos_stopped         true when the row ended on an EOS token
    finish_reason       "stop" (EOS) or "length"
    text                decode(generated_ids), special tokens skipped
    head_text           decode of the first 16 generated tokens

HFReference: transformers model in float16 on cuda, eval mode, its own greedy loop with
past_key_values (argmax every step). Requests are grouped by exact prompt token length (and
max_tokens / ignore_eos) and each group runs in batches of at most 32 rows: no padding, no
attention mask. ignore_eos rows never stop and generate exactly max_tokens tokens. Other rows
stop at their first EOS (generation_config eos_token_id list); the batch keeps going until
every row stopped or max_tokens, and each row is truncated at its own stop.
"""

import argparse
import json
import sys
import time

from . import fakemodel
from .tokenizer import MockTokenizer, strip_incomplete_utf8

HEAD_TOKENS = 16
MAX_BATCH = 32


class ReferenceMismatch(Exception):
    pass


def finalize(tok, prompt_tokens, generated_ids, eos_stopped):
    text = strip_incomplete_utf8(tok.decode(generated_ids))
    head = strip_incomplete_utf8(tok.decode(generated_ids[:HEAD_TOKENS]))
    if not text.startswith(head):
        raise ReferenceMismatch("decoded head is not a prefix of the decoded text: %r vs %r" % (head, text[:200]))
    return {
        "prompt_tokens": prompt_tokens,
        "generated_ids": list(generated_ids),
        "generated_tokens": len(generated_ids) + (1 if eos_stopped else 0),
        "eos_stopped": bool(eos_stopped),
        "finish_reason": "stop" if eos_stopped else "length",
        "text": text,
        "head_text": head,
    }


class MockReference:
    """Reproduces grader.fakemodel, i.e. exactly what a correct mock_engine returns."""

    def __init__(self):
        self.tok = MockTokenizer()

    def run(self, requests, log=None):
        results = {}
        for r in requests:
            if r["kind"] == "completion":
                context = r["prompt"].split()
            else:
                context = fakemodel.chat_template_words(r["messages"])
            words, eos_stopped = fakemodel.generate(context, int(r["max_tokens"]), bool(r["ignore_eos"]))
            ids = [self.tok.encode(w)[0] if w != fakemodel.EOS else self.tok.eos_id for w in words]
            results[r["id"]] = finalize(self.tok, len(context), ids, eos_stopped)
        return results, {"provider": "mock"}


class HFReference:
    def __init__(self, model_dir, device="cuda", dtype="float16", max_batch=MAX_BATCH):
        import torch  # real path only
        import transformers
        from transformers import AutoModelForCausalLM

        from .tokenizer import HFTokenizer

        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ReferenceMismatch("CUDA is not available")
        torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
        self.tok = HFTokenizer(model_dir)
        self.eos = set(self.tok.eos_token_ids)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch_dtype, attn_implementation="sdpa", local_files_only=True)
        except TypeError:  # older transformers without the dtype= keyword
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch_dtype, attn_implementation="sdpa", local_files_only=True)
        model = model.to(dtype=torch_dtype).to(device).eval()
        bad = {str(p.dtype) for p in model.parameters() if p.dtype != torch_dtype}
        if bad:
            raise ReferenceMismatch("model parameters not in %s: %s" % (dtype, sorted(bad)))
        self.model = model
        # Only the last position's logits are computed: full-sequence logits for 32 x 1024
        # prompt tokens over a 152k vocabulary would not fit on a T4.
        self.body = model.base_model
        self.lm_head = model.get_output_embeddings()
        self.device = device
        self.max_batch = max_batch
        self.meta = {
            "provider": "hf",
            "model_dir": model_dir,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
            "dtype": dtype,
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "eos_token_ids": sorted(self.eos),
            "max_batch": max_batch,
        }

    def _prompt_ids(self, r):
        if r["kind"] == "completion":
            return self.tok.encode(r["prompt"])
        return self.tok.chat_prompt_ids(r["messages"])

    def _generate(self, batch_ids, max_tokens, ignore_eos):
        torch = self.torch
        rows = len(batch_ids)
        out = [[] for _ in range(rows)]
        stopped = [False] * rows
        step_input = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        past = None
        with torch.inference_mode():
            for _ in range(max_tokens):
                hidden = self.body(input_ids=step_input, past_key_values=past, use_cache=True)
                past = hidden.past_key_values
                logits = self.lm_head(hidden.last_hidden_state[:, -1, :])
                next_ids = torch.argmax(logits, dim=-1)
                for i, t in enumerate(next_ids.tolist()):
                    if stopped[i]:
                        continue
                    if not ignore_eos and t in self.eos:
                        stopped[i] = True
                    else:
                        out[i].append(t)
                if all(stopped):
                    break
                step_input = next_ids[:, None]
        return out, stopped

    def run(self, requests, log=None):
        t0 = time.perf_counter()
        prompt_ids = {r["id"]: self._prompt_ids(r) for r in requests}
        groups = {}
        for r in sorted(requests, key=lambda r: r["id"]):
            key = (len(prompt_ids[r["id"]]), int(r["max_tokens"]), bool(r["ignore_eos"]))
            groups.setdefault(key, []).append(r)
        batches = []
        for key in sorted(groups):
            members = groups[key]
            for i in range(0, len(members), self.max_batch):
                batches.append((key, members[i:i + self.max_batch]))
        results = {}
        for n, ((length, max_tokens, ignore_eos), members) in enumerate(batches):
            ids = [prompt_ids[r["id"]] for r in members]
            gen, stopped = self._generate(ids, max_tokens, ignore_eos)
            for r, g, s in zip(members, gen, stopped):
                results[r["id"]] = finalize(self.tok, length, g, s)
            if log:
                log("batch %d/%d: %d rows, prompt %d tokens, %.1f s elapsed"
                    % (n + 1, len(batches), len(members), length, time.perf_counter() - t0))
        meta = dict(self.meta, batches=len(batches), requests=len(requests), elapsed_s=time.perf_counter() - t0)
        return results, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--provider", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model-dir")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH)
    args = ap.parse_args(argv)

    def log(msg):
        print("[reference] " + msg, file=sys.stderr, flush=True)

    with open(args.requests, encoding="utf-8") as f:
        requests = json.load(f)
    t0 = time.perf_counter()
    if args.provider == "mock":
        provider = MockReference()
    else:
        if not args.model_dir:
            ap.error("--model-dir is required for --provider hf")
        provider = HFReference(args.model_dir, device=args.device, dtype=args.dtype, max_batch=args.max_batch)
        log("model loaded in %.1f s" % (time.perf_counter() - t0))
    results, meta = provider.run(requests, log=log)
    meta["total_s"] = time.perf_counter() - t0
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f)
    log("wrote %d results in %.1f s" % (len(results), meta["total_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

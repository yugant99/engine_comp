"""HFReference on a tiny random Qwen2 model (cpu, float32). Skipped without torch/transformers.

The batched, KV-cached greedy loop of grader.reference must give exactly what a naive loop
gives: one request at a time, full recompute of the whole sequence every step, logits from the
model's own forward. This covers batching, past_key_values, last-position logits, per-row EOS
truncation, ignore_eos and the chat-template path, without a GPU or the real model.
"""

import json
import os
import subprocess

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from conftest import PY, ROOT  # noqa: E402

WORDS = ["w%d" % i for i in range(120)]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"]
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }}<|im_end|> {% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")


def naive_greedy(model, prompt_ids, max_tokens, ignore_eos, eos):
    ids = list(prompt_ids)
    out = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            t = int(torch.argmax(logits))
            if not ignore_eos and t in eos:
                return out, True
            out.append(t)
            ids.append(t)
    return out, False


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    path = str(tmp_path_factory.mktemp("tiny-qwen2"))
    vocab = {w: i for i, w in enumerate(SPECIALS + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.add_special_tokens(SPECIALS)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                     unk_token="<unk>")
    hf_tok.chat_template = TEMPLATE
    hf_tok.save_pretrained(path)

    # Tied embeddings as in Qwen2.5-1.5B; a large init so the random model does not just repeat
    # one token (and near-ties in the logits are unlikely).
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True, initializer_range=0.3)
    model = Qwen2ForCausalLM(cfg).eval()
    model.save_pretrained(path)

    # Pick a second EOS id that the model actually emits at different steps for different
    # prompts, so rows of one batch stop at different times.
    prompts = [" ".join(WORDS[(7 * i + j) % len(WORDS)] for j in range(10)) for i in range(12)]
    first_pos = {}
    for p in prompts:
        gen, _ = naive_greedy(model, hf_tok.encode(p, add_special_tokens=False), 24, True, set())
        for pos, t in enumerate(gen):
            first_pos.setdefault(t, {})
            first_pos[t].setdefault(p, pos)
    best = max(first_pos, key=lambda t: (len(set(first_pos[t].values())), len(first_pos[t])))
    eos = [vocab["<|im_end|>"], best]
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump({"eos_token_id": eos, "bos_token_id": vocab["<|endoftext|>"]}, f)
    return path, model, set(eos), prompts


def requests_for(prompts):
    reqs = []
    for i, p in enumerate(prompts):
        reqs.append({"id": "nat-%02d" % i, "kind": "completion", "prompt": p, "max_tokens": 24, "ignore_eos": False})
        reqs.append({"id": "ign-%02d" % i, "kind": "completion", "prompt": p, "max_tokens": 20, "ignore_eos": True})
    short = [" ".join(WORDS[(3 * i + j) % len(WORDS)] for j in range(6)) for i in range(5)]
    for i, p in enumerate(short):
        reqs.append({"id": "short-%02d" % i, "kind": "completion", "prompt": p, "max_tokens": 16, "ignore_eos": False})
    for i in range(4):
        msgs = [{"role": "system", "content": "w1 w2 w3"}, {"role": "user", "content": "w%d w%d" % (i, i + 9)}]
        reqs.append({"id": "chat-%02d" % i, "kind": "chat", "messages": msgs, "max_tokens": 24, "ignore_eos": False})
    return reqs


def test_batched_reference_matches_naive_loop(tiny_model):
    from grader.reference import HFReference

    path, model, eos, prompts = tiny_model
    ref = HFReference(path, device="cpu", dtype="float32")
    reqs = requests_for(prompts)
    results, meta = ref.run(reqs)
    assert meta["batches"] < len(reqs)  # equal-length prompts really were batched
    tok = ref.tok
    stopped_at = set()
    for r in reqs:
        ids = tok.encode(r["prompt"]) if r["kind"] == "completion" else tok.chat_prompt_ids(r["messages"])
        want, want_eos = naive_greedy(model, ids, r["max_tokens"], r["ignore_eos"], eos)
        got = results[r["id"]]
        assert got["prompt_tokens"] == len(ids)
        assert got["generated_ids"] == want, r["id"]
        assert got["eos_stopped"] == want_eos, r["id"]
        assert got["generated_tokens"] == len(want) + (1 if want_eos else 0)
        assert got["text"] == tok.decode(want)
        assert got["text"].startswith(got["head_text"])
        if r["ignore_eos"]:
            assert len(got["generated_ids"]) == r["max_tokens"] and not got["eos_stopped"]
        if want_eos:
            stopped_at.add(len(want))
    # The fixture's EOS choice must make rows stop, at more than one step.
    assert len(stopped_at) >= 2, stopped_at


def test_chat_prompt_matches_apply_chat_template(tiny_model):
    from transformers import AutoTokenizer

    from grader.tokenizer import HFTokenizer

    path = tiny_model[0]
    msgs = [{"role": "system", "content": "w1 w2"}, {"role": "user", "content": "w5 w6 w7"}]
    ours = HFTokenizer(path).chat_prompt_ids(msgs)
    theirs = AutoTokenizer.from_pretrained(path).apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(theirs, dict) or hasattr(theirs, "keys"):
        theirs = theirs["input_ids"]
    assert ours == list(theirs)


def test_reference_cli_subprocess(tiny_model, tmp_path):
    path, _, _, prompts = tiny_model
    reqs = requests_for(prompts)[:6]
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps(reqs))
    p = subprocess.run([PY, "-m", "grader.reference", "--provider", "hf", "--model-dir", path, "--requests", str(inp),
                        "--out", str(out), "--device", "cpu", "--dtype", "float32"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    data = json.loads(out.read_text())
    assert set(data["results"]) == {r["id"] for r in reqs}
    assert data["meta"]["provider"] == "hf" and data["meta"]["dtype"] == "float32"

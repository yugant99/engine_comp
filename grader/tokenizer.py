"""The one tokenizer interface the grader uses.

    encode(text)            -> list[int]   no template, no added special tokens
    encode_batch(texts)     -> list[list[int]]
    decode(ids)             -> str         special tokens skipped
    chat_prompt_ids(msgs)   -> list[int]   the model's chat template, generation prompt added
    eos_token_ids           -> list[int]

HFTokenizer is the real one (transformers AutoTokenizer, imported lazily). MockTokenizer is a
whitespace tokenizer that matches grader.fakemodel, for local tests without any ML library.
"""

import json
import os

from . import fakemodel


def strip_incomplete_utf8(text):
    """Drop U+FFFD characters at the end of a decoded string.

    A decode that stops in the middle of a multi-byte UTF-8 character ends in U+FFFD. Engines hold
    incomplete sequences until they complete (SPEC section 4), so such a tail is never emitted;
    the reference text must not contain it either.
    """
    return text.rstrip("�")


def read_eos_token_ids(model_dir):
    """eos_token_id from generation_config.json (int or list), as a list."""
    path = os.path.join(model_dir, "generation_config.json")
    with open(path, encoding="utf-8") as f:
        value = json.load(f).get("eos_token_id")
    if value is None:
        raise ValueError("generation_config.json has no eos_token_id")
    return [int(v) for v in (value if isinstance(value, list) else [value])]


class HFTokenizer:
    kind = "hf"

    def __init__(self, model_dir):
        from transformers import AutoTokenizer  # real path only

        self._tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        # The corpus is tokenized in one call; avoid the "sequence longer than max length" warning.
        self._tok.model_max_length = 10**9
        self.eos_token_ids = read_eos_token_ids(model_dir)

    def encode(self, text):
        return list(self._tok.encode(text, add_special_tokens=False))

    def encode_batch(self, texts):
        if not texts:
            return []
        return [list(ids) for ids in self._tok(list(texts), add_special_tokens=False)["input_ids"]]

    def decode(self, ids):
        return self._tok.decode(list(ids), skip_special_tokens=True)

    def chat_prompt_text(self, messages):
        # Render to text, then tokenize without special tokens; this is what vLLM does and it
        # avoids version differences in apply_chat_template(tokenize=True) return types.
        text = self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if not isinstance(text, str):
            raise TypeError("apply_chat_template(tokenize=False) returned %r" % type(text))
        return text

    def chat_prompt_ids(self, messages):
        return self.encode(self.chat_prompt_text(messages))


class MockTokenizer:
    """Whitespace tokenizer. Every token decodes to " " + word; EOS decodes to ""."""

    kind = "mock"

    def __init__(self):
        self._id_of = {}
        self._word_of = []
        for w in fakemodel.VOCAB:
            self._intern(w)
        self.eos_id = self._intern(fakemodel.EOS)
        self.eos_token_ids = [self.eos_id]

    def _intern(self, word):
        i = self._id_of.get(word)
        if i is None:
            i = len(self._word_of)
            self._id_of[word] = i
            self._word_of.append(word)
        return i

    def encode(self, text):
        return [self._intern(w) for w in text.split()]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        return "".join(fakemodel.token_text(self._word_of[i]) for i in ids)

    def chat_prompt_text(self, messages):
        return " ".join(fakemodel.chat_template_words(messages))

    def chat_prompt_ids(self, messages):
        return self.encode(self.chat_prompt_text(messages))


def make_tokenizer(provider, model_dir=None):
    if provider == "mock":
        return MockTokenizer()
    if provider == "hf":
        if not model_dir:
            raise ValueError("the hf tokenizer needs --model-dir")
        return HFTokenizer(model_dir)
    raise ValueError("unknown tokenizer provider %r" % provider)

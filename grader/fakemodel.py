"""A deterministic fake language model over whitespace "tokens" (words).

Used only for local tests: mock_engine serves it over HTTP and MockReference computes the
expected outputs with the same code, so a correct mock engine matches the reference exactly.

A token is a word. Generated text is the concatenation of " " + word for every generated
token, and the end-of-sequence token EOS decodes to the empty string. The next greedy token
is a hash of everything seen so far, so different prompts give different continuations
(a cross-request mix-up is detectable) and the same prompt always gives the same one.

Hashes use zlib.crc32 and integer arithmetic only, never Python's salted hash(), so two
processes always agree.
"""

import random
import zlib

EOS = "<eos>"
DEFAULT_SYSTEM = "You are a helpful assistant."
MASK64 = (1 << 64) - 1

# Plain words plus some multi-byte UTF-8 ones, so streaming clients see characters that a
# TCP packet boundary can split.
_SYLLABLES = ["ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "zu", "pe", "do", "fa"]
VOCAB = [a + b for a in _SYLLABLES for b in _SYLLABLES] + [
    "café", "naïve", "über", "Ωmega", "日本語", "Привет", "😀", "façade", "jalapeño", "ångström",
]
EOS_EVERY = 23  # roughly one greedy step in 23 is an EOS

FORCED_PREFIX = ["Reply", "with", "only", "the", "word:"]


def _word_hash(word):
    return zlib.crc32(word.encode("utf-8"))


def _mix(x):
    """splitmix64 finaliser: spreads the bits of the running state."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def absorb(state, word):
    return ((state * 0x100000001B3) ^ _word_hash(word)) & MASK64


def initial_state(words):
    state = 0xCBF29CE484222325
    for w in words:
        state = absorb(state, w)
    return state


def greedy_token(state):
    z = _mix(state)
    if z % EOS_EVERY == 0:
        return EOS
    return VOCAB[(z >> 8) % len(VOCAB)]


def chat_template_words(messages):
    """The mock chat template, as a list of whitespace tokens, generation prompt included."""
    words = []
    if not messages or messages[0].get("role") != "system":
        words += ["<system>"] + DEFAULT_SYSTEM.split() + ["</system>"]
    for m in messages:
        role = str(m.get("role"))
        words += ["<%s>" % role] + str(m.get("content", "")).split() + ["</%s>" % role]
    words.append("<assistant>")
    return words


def _forced_reply(context):
    """Instruction-following stand-in: "Reply with only the word: X" makes the model say X, then EOS."""
    if len(context) < 3 or context[-1] != "<assistant>" or context[-2] != "</user>":
        return None
    try:
        start = len(context) - 1 - context[::-1].index("<user>")
    except ValueError:
        return None
    user = context[start + 1:-2]
    if user[:5] == FORCED_PREFIX and len(user) >= 6:
        return [user[5].rstrip(".")]
    return None


class Generation:
    """Step-by-step generation for one request.

    next_token() returns the token the model picks now; advance(token) feeds it back.
    With temperature > 0 the token is drawn from a PRNG seeded by (seed, state), so the same
    seed gives the same sample; top_p below 0.05 collapses to the greedy token.
    """

    def __init__(self, context_words, temperature=0.0, top_p=1.0, seed=None):
        self.state = initial_state(context_words)
        self.forced = _forced_reply(context_words)
        self.steps = 0
        self.temperature = float(temperature or 0.0)
        self.top_p = 1.0 if top_p is None else float(top_p)
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1 << 62)

    def next_token(self):
        if self.forced is not None and self.steps <= len(self.forced):
            return self.forced[self.steps] if self.steps < len(self.forced) else EOS
        greedy = greedy_token(self.state)
        if self.temperature <= 0 or self.top_p < 0.05:
            return greedy
        rng = random.Random("%d:%d" % (self.seed, self.state))
        if rng.random() < 0.5:
            return greedy
        return VOCAB[rng.randrange(len(VOCAB))]

    def advance(self, token):
        self.state = absorb(self.state, token)
        self.steps += 1


def generate(context_words, max_tokens, ignore_eos):
    """Greedy reference generation.

    Returns (tokens, eos_stopped). tokens excludes the terminating EOS; with ignore_eos the
    EOS tokens the model picks are kept in the list (they decode to "").
    """
    gen = Generation(context_words)
    tokens = []
    for _ in range(max_tokens):
        tok = gen.next_token()
        if tok == EOS and not ignore_eos:
            return tokens, True
        tokens.append(tok)
        gen.advance(tok)
    return tokens, False


def token_text(token):
    return "" if token == EOS else " " + token

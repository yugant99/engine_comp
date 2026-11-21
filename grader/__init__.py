"""Engine Arena grader (spec v1).

Run from the engine-arena directory:

    python -m grader.grade --engine-dir DIR --model-dir DIR --corpus FILE --out result.json

The mock path (``--reference mock``) needs only the standard library and aiohttp.
torch and transformers are imported lazily, only on the real (``hf``) path.
"""

SPEC_VERSION = "1"
GRADER_VERSION = "1.0.0"

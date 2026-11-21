"""Banned-dependency scan (SPEC section 2).

A *finding* fails the run; a *warning* is only reported. Findings use exact matches on
normalised names, to keep false positives (which would score a team 0) out:

  - imports in every .py file (ast; also importlib.import_module("x") / __import__("x") with a
    literal name); the top-level module name must be in BANNED_MODULES
  - requirement names in requirements*.txt (following -r/-c), pyproject.toml, setup.py,
    setup.cfg, and crate names in Cargo.toml
  - git submodules (.gitmodules) whose repository name is a banned project
  - directories whose normalised name is a banned project ("vendored" copies)
  - distributions installed in the engine's own environment (its virtualenv, or user
    site-packages under its HOME)

Warnings: shell-script lines that install or clone something with a banned name, directory names
that merely start with a banned name, and banned distributions visible from the system
site-packages (the base image), which the engine did not install itself.

    python -m grader.scan --engine-dir DIR [--pip-list FILE]
"""

import argparse
import ast
import configparser
import json
import os
import re
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

# Top-level import names.
BANNED_MODULES = {
    "vllm": "vLLM", "vllm_flash_attn": "vLLM",
    "nanovllm": "nano-vllm", "nano_vllm": "nano-vllm",
    "sglang": "SGLang", "sgl_kernel": "SGLang",
    "llama_cpp": "llama-cpp-python",
    "text_generation_server": "Text Generation Inference",
    "tensorrt": "TensorRT", "tensorrt_llm": "TensorRT-LLM", "torch_tensorrt": "TensorRT",
    "onnxruntime": "ONNX Runtime",
    "ctranslate2": "CTranslate2",
    "exllama": "ExLlama", "exllamav2": "ExLlama", "exllamav3": "ExLlama",
    "mlc_llm": "MLC", "mlc_chat": "MLC",
    "lmdeploy": "LMDeploy",
    "lightllm": "LightLLM",
    "mistralrs": "mistral.rs",
    "candle": "candle",
    "ollama": "Ollama",
    "transformers": "transformers",
    "optimum": "optimum",
    "accelerate": "accelerate",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
    "xformers": "xformers",
}

# Distribution names, normalised as in PEP 503 (lower case, runs of -_. become -).
BANNED_DISTS = {
    "vllm": "vLLM", "vllm-flash-attn": "vLLM",
    "nano-vllm": "nano-vllm", "nanovllm": "nano-vllm",
    "sglang": "SGLang", "sgl-kernel": "SGLang",
    "llama-cpp-python": "llama-cpp-python",
    "text-generation-server": "Text Generation Inference",
    "tensorrt": "TensorRT", "tensorrt-cu11": "TensorRT", "tensorrt-cu12": "TensorRT",
    "tensorrt-cu13": "TensorRT", "tensorrt-libs": "TensorRT", "tensorrt-bindings": "TensorRT",
    "nvidia-tensorrt": "TensorRT", "torch-tensorrt": "TensorRT", "tensorrt-llm": "TensorRT-LLM",
    "onnxruntime": "ONNX Runtime", "onnxruntime-gpu": "ONNX Runtime",
    "ctranslate2": "CTranslate2",
    "exllama": "ExLlama", "exllamav2": "ExLlama", "exllamav3": "ExLlama",
    "mlc-llm": "MLC", "mlc-chat": "MLC", "mlc-ai": "MLC",
    "lmdeploy": "LMDeploy",
    "lightllm": "LightLLM",
    "mistralrs": "mistral.rs", "mistralrs-cuda": "mistral.rs",
    "candle": "candle",
    "ollama": "Ollama",
    "transformers": "transformers",
    "optimum": "optimum", "optimum-nvidia": "optimum", "optimum-onnx": "optimum",
    "accelerate": "accelerate",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
    "xformers": "xformers",
}

BANNED_CRATES = {
    "candle-core": "candle", "candle-nn": "candle", "candle-transformers": "candle",
    "candle-flash-attn": "candle", "candle-kernels": "candle",
    "mistralrs": "mistral.rs", "mistralrs-core": "mistral.rs",
    "llama-cpp-2": "llama.cpp", "llama-cpp-sys-2": "llama.cpp", "llama_cpp": "llama.cpp",
    "ort": "ONNX Runtime", "ort-sys": "ONNX Runtime", "ct2rs": "CTranslate2",
}

# Directory / repository names, normalised by dropping case and "-", "_", ".".
BANNED_DIR_NAMES = {
    "vllm": "vLLM", "nanovllm": "nano-vllm", "sglang": "SGLang", "sglkernel": "SGLang",
    "llamacpp": "llama.cpp", "llamacpppython": "llama-cpp-python",
    "textgenerationinference": "Text Generation Inference",
    "tensorrt": "TensorRT", "tensorrtllm": "TensorRT-LLM", "onnxruntime": "ONNX Runtime",
    "ctranslate2": "CTranslate2", "exllama": "ExLlama", "exllamav2": "ExLlama", "exllamav3": "ExLlama",
    "mlcllm": "MLC", "lmdeploy": "LMDeploy", "lightllm": "LightLLM", "mistralrs": "mistral.rs",
    "candle": "candle", "ollama": "Ollama", "transformers": "transformers", "optimum": "optimum",
    "accelerate": "accelerate", "peft": "peft", "bitsandbytes": "bitsandbytes", "xformers": "xformers",
}
# Distinctive names: a directory that starts with one of these (e.g. "vllm-main") is a warning.
DISTINCTIVE_PREFIXES = ("vllm", "nanovllm", "sglang", "llamacpp", "textgenerationinference",
                        "tensorrtllm", "onnxruntime", "ctranslate2", "exllama", "mlcllm", "lmdeploy",
                        "lightllm", "mistralrs", "xformers", "bitsandbytes")

SKIP_DIRS = {".git"}


def normalize_dist(name):
    return re.sub(r"[-_.]+", "-", name).lower().strip("-")


def normalize_dir(name):
    return re.sub(r"[-_.]+", "", name).lower()


def _rel(path, root):
    return os.path.relpath(path, root)


# ---------------------------------------------------------------- python imports

def python_imports(source):
    """Top-level module names imported by a Python source string."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _regex_imports(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if fname in ("import_module", "__import__") and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and not a.value.startswith("."):
                    names.add(a.value.split(".")[0])
    return names


def _regex_imports(source):
    names = set()
    for m in re.finditer(r"^\s*(?:from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*))", source, re.M):
        names.add(m.group(1) or m.group(2))
    return names


# ---------------------------------------------------------------- requirement files

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def requirement_name(spec):
    """The distribution name of a PEP 508 requirement string (or None)."""
    spec = spec.strip()
    if not spec or spec.startswith("#"):
        return None
    m = re.search(r"#egg=([A-Za-z0-9._-]+)", spec)
    if m:
        return m.group(1)
    m = _REQ_NAME.match(spec)
    return m.group(1) if m else None


def requirements_file_names(path, seen=None):
    """Names in a requirements file, following -r/-c includes."""
    seen = set() if seen is None else seen
    path = os.path.abspath(path)
    if path in seen or not os.path.isfile(path):
        return []
    seen.add(path)
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split(" #")[0].strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(-r|-c|--requirement|--constraint)[\s=]+(\S+)", line)
            if m:
                out += requirements_file_names(os.path.join(os.path.dirname(path), m.group(2)), seen)
                continue
            if line.startswith("-e ") or line.startswith("--editable"):
                line = line.split(None, 1)[1] if " " in line else ""
            elif line.startswith("-"):
                continue
            name = requirement_name(line)
            if name:
                out.append(name)
            # A direct URL: also look at the repository name, e.g. git+https://.../vllm.git
            for url in re.findall(r"(?:git\+)?https?://\S+", line):
                repo = url.split("#")[0].split("@")[0].rstrip("/").split("/")[-1]
                repo = repo[:-4] if repo.endswith(".git") else repo
                out.append(repo)
    return out


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _strings(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _strings(x)


def pyproject_names(path):
    if tomllib is None:
        return []
    with open(path, "rb") as f:
        data = tomllib.load(f)
    specs = []
    project = data.get("project", {})
    specs += list(_strings(project.get("dependencies", [])))
    specs += list(_strings(project.get("optional-dependencies", {})))
    specs += list(_strings(data.get("build-system", {}).get("requires", [])))
    specs += list(_strings(data.get("dependency-groups", {})))
    poetry = data.get("tool", {}).get("poetry", {})
    names = [requirement_name(s) for s in specs]
    for table in ("dependencies", "dev-dependencies"):
        names += list(poetry.get(table, {}).keys())
    for group in poetry.get("group", {}).values():
        names += list(group.get("dependencies", {}).keys())
    return [n for n in names if n]


def setup_py_names(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in ("install_requires", "setup_requires",
                                                          "extras_require", "tests_require"):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    names.append(requirement_name(sub.value))
    return [n for n in names if n]


def setup_cfg_names(path):
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error:
        return []
    names = []
    for section in cp.sections():
        for key, value in cp.items(section, raw=True):
            if section == "options" and key not in ("install_requires", "setup_requires", "tests_require"):
                continue
            if section != "options" and section != "options.extras_require":
                continue
            for line in value.splitlines():
                names.append(requirement_name(line))
    return [n for n in names if n]


def cargo_names(path):
    if tomllib is None:
        return []
    with open(path, "rb") as f:
        data = tomllib.load(f)
    names = []

    def tables(d):
        for key in ("dependencies", "dev-dependencies", "build-dependencies"):
            if isinstance(d.get(key), dict):
                yield d[key]

    names += [n for t in tables(data) for n in t]
    names += [n for t in tables(data.get("workspace", {})) for n in t]
    for target in data.get("target", {}).values():
        if isinstance(target, dict):
            names += [n for t in tables(target) for n in t]
    # A renamed dependency: foo = { package = "candle-core", ... }
    for t in list(tables(data)) + list(tables(data.get("workspace", {}))):
        for v in t.values():
            if isinstance(v, dict) and isinstance(v.get("package"), str):
                names.append(v["package"])
    return names


def gitmodules_repos(path):
    repos = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*url\s*=\s*(\S+)", line)
            if m:
                repo = m.group(1).rstrip("/").split("/")[-1].split(":")[-1]
                repos.append(repo[:-4] if repo.endswith(".git") else repo)
    return repos


_SHELL_ACTION = re.compile(r"\b(pip3?|uv|conda|mamba|git|curl|wget|cargo|npm)\b.*\b(install|add|clone|get|https?://)", re.I)


def shell_warnings(path, root):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            code = line.split("#")[0]
            if not _SHELL_ACTION.search(code):
                continue
            words = {normalize_dir(w) for w in re.findall(r"[A-Za-z0-9._-]+", code)}
            words |= {normalize_dist(w) for w in re.findall(r"[A-Za-z0-9._-]+", code)}
            hits = sorted({BANNED_DIR_NAMES.get(w) or BANNED_DISTS.get(w) for w in words} - {None})
            if hits:
                out.append("%s:%d installs or fetches something named like %s: %s"
                           % (_rel(path, root), lineno, ", ".join(hits), line.strip()[:200]))
    return out


# ---------------------------------------------------------------- the source tree

def is_test_file(rel_path):
    parts = rel_path.replace(os.sep, "/").split("/")
    name = parts[-1]
    return (any(p in ("tests", "test") for p in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def scan_source_tree(root, skip_paths=(), exempt_tests=False):
    """Static scan of an engine directory. Returns (findings, warnings).

    exempt_tests: banned imports in test files (under tests/ or test/, test_*.py, *_test.py,
    conftest.py) become warnings. Off by default: SPEC section 2 has no such exception, and
    run.sh could import engine code from a test file.
    """
    root = os.path.abspath(root)
    skip = {os.path.abspath(p) for p in skip_paths}
    findings, warnings = [], []

    def dist_hits(names, where):
        for n in names:
            project = BANNED_DISTS.get(normalize_dist(n)) or BANNED_DIR_NAMES.get(normalize_dir(n))
            if project:
                findings.append("%s: depends on %r (%s)" % (where, n, project))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and os.path.join(dirpath, d) not in skip)
        for d in dirnames:
            nd = normalize_dir(d)
            rel = _rel(os.path.join(dirpath, d), root)
            if nd in BANNED_DIR_NAMES:
                findings.append("%s/: directory named after banned project %s" % (rel, BANNED_DIR_NAMES[nd]))
            elif any(nd.startswith(p) for p in DISTINCTIVE_PREFIXES):
                warnings.append("%s/: directory name starts with a banned project name" % rel)
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            rel = _rel(path, root)
            low = fn.lower()
            try:
                if low.endswith(".py"):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        source = f.read()
                    for mod in sorted(python_imports(source)):
                        if mod in BANNED_MODULES:
                            msg = "%s: imports %s (%s)" % (rel, mod, BANNED_MODULES[mod])
                            if exempt_tests and is_test_file(rel):
                                warnings.append(msg + " [test file, exempt]")
                            else:
                                findings.append(msg)
                    if low == "setup.py":
                        dist_hits(setup_py_names(source), rel)
                elif low.startswith("requirements") and low.endswith((".txt", ".in")):
                    dist_hits(requirements_file_names(path), rel)
                elif low == "pyproject.toml":
                    dist_hits(pyproject_names(path), rel)
                elif low == "setup.cfg":
                    dist_hits(setup_cfg_names(path), rel)
                elif low == "cargo.toml":
                    for n in cargo_names(path):
                        if n in BANNED_CRATES:
                            findings.append("%s: depends on crate %r (%s)" % (rel, n, BANNED_CRATES[n]))
                elif low == ".gitmodules":
                    for repo in gitmodules_repos(path):
                        project = BANNED_DIR_NAMES.get(normalize_dir(repo))
                        if project:
                            findings.append("%s: git submodule %r (%s)" % (rel, repo, project))
                if low.endswith((".sh", ".bash")) or fn in ("run.sh", "build.sh", "Makefile", "Dockerfile"):
                    warnings += shell_warnings(path, root)
            except Exception as e:  # an unreadable file must not abort the scan
                warnings.append("%s: could not scan (%s: %s)" % (rel, type(e).__name__, e))
    return findings, warnings


# ---------------------------------------------------------------- installed packages

def parse_pip_list(text):
    """(name, version) pairs from `pip list` / `uv pip list` output: JSON, freeze or columns."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        return [(d["name"], d.get("version")) for d in json.loads(text)]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("Package", "---", "Using ", "#")):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            out.append((name.strip(), version.strip()))
        elif " @ " in line:
            out.append((line.split(" @ ", 1)[0].strip(), None))
        else:
            parts = line.split()
            out.append((parts[0], parts[1] if len(parts) > 1 else None))
    return out


def scan_pip_list(text):
    """Findings for a pip-list output that lists only the engine's own environment."""
    return ["installed package %s %s (%s)" % (n, v or "", BANNED_DISTS[normalize_dist(n)])
            for n, v in parse_pip_list(text) if normalize_dist(n) in BANNED_DISTS]


def scan_installed(dists, local_roots):
    """dists: [{"name", "version", "location"}] as seen by the engine's Python.

    A banned distribution under one of local_roots (the venv, the engine's HOME) is a finding;
    one elsewhere (system site-packages of the base image) is a warning.
    """
    roots = [os.path.realpath(r) + os.sep for r in local_roots]
    findings, warnings = [], []
    for d in dists:
        name = d.get("name") or ""
        project = BANNED_DISTS.get(normalize_dist(name))
        if not project:
            continue
        loc = os.path.realpath(d.get("location") or "")
        msg = "%s %s (%s) at %s" % (name, d.get("version") or "", project, d.get("location"))
        if any((loc + os.sep).startswith(r) for r in roots):
            findings.append("installed in the engine environment: " + msg)
        else:
            warnings.append("visible from system site-packages (not installed by the engine): " + msg)
    return findings, warnings


# Run with the engine's own Python to list what it can import.
LIST_DISTS_SCRIPT = r"""
import importlib.metadata as m, json, sys
out = []
for d in m.distributions():
    try:
        name = d.metadata["Name"]
    except Exception:
        name = None
    out.append({"name": name, "version": d.version, "location": str(getattr(d, "_path", "") or "")})
print("ARENA-DISTS " + json.dumps({"dists": out, "sys_path": sys.path}))
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Banned-dependency scan of an engine directory.")
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--pip-list", help="file with `uv pip list` output of the engine's virtualenv")
    args = ap.parse_args(argv)
    findings, warnings = scan_source_tree(args.engine_dir)
    if args.pip_list:
        with open(args.pip_list, encoding="utf-8") as f:
            findings += scan_pip_list(f.read())
    for w in warnings:
        print("warning: " + w)
    for x in findings:
        print("FINDING: " + x)
    print("%d finding(s), %d warning(s)" % (len(findings), len(warnings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

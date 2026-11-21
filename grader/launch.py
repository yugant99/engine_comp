"""Install, build, start and stop an engine (SPEC section 3).

    uv venv --system-site-packages        virtualenv that sees the base image
    uv pip install -r requirements.txt    if present, 10-minute limit
    bash build.sh                         if present, 10-minute limit
    bash run.sh                           env MODEL_DIR, HOST, PORT, HF_HUB_OFFLINE=1, no proxies

Every engine-side command runs, if requested, as another Unix user (--run-as: setpriv when
available, else sudo -n) and with a fresh HOME inside the work directory. run.sh additionally
runs under `taskset -c CORES` when cores are given, in its own session / process group, with
stdout and stderr appended to the engine log. Stopping sends SIGTERM to the group and SIGKILL
after the grace period; with --run-as every remaining process of that user is killed too.
Each optional piece (run-as, taskset) is skipped when not requested, so this also works
unprivileged on a Mac.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from .scan import LIST_DISTS_SCRIPT

INSTALL_TIMEOUT_S = 600
BUILD_TIMEOUT_S = 600
VENV_TIMEOUT_S = 120

PROXY_VARS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy", "socks_proxy"}
# Never hand these to untrusted engine code.
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "API_KEY", "ACCESS_KEY", "PRIVATE_KEY")
_DROP_EXACT = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "SSH_AUTH_SOCK", "VIRTUAL_ENV", "CONDA_PREFIX",
               # These would point uv at another environment, e.g. the base image (Colab sets the first).
               "UV_SYSTEM_PYTHON", "UV_PYTHON", "UV_BREAK_SYSTEM_PACKAGES", "UV_PROJECT_ENVIRONMENT",
               "PIP_TARGET", "PIP_PREFIX", "PIP_USER"}
_DROP_PREFIXES = ("AWS_", "GITHUB_", "GH_", "GOOGLE_", "AZURE_", "SUDO_", "ARENA_")


class LaunchError(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def engine_base_env(environ=None):
    environ = os.environ if environ is None else environ
    env = {}
    for k, v in environ.items():
        if k.lower() in PROXY_VARS or k in _DROP_EXACT or k.startswith(_DROP_PREFIXES):
            continue
        if any(m in k.upper() for m in _SECRET_MARKERS):
            continue
        env[k] = v
    return env


class Launcher:
    def __init__(self, engine_dir, model_dir, work_dir, log_path, host="127.0.0.1", port=8000,
                 cores=None, run_as=None, python=None, stop_grace_s=10.0):
        self.engine_dir = os.path.abspath(engine_dir)
        self.model_dir = os.path.abspath(model_dir) if model_dir else ""
        self.work_dir = os.path.abspath(work_dir)
        self.log_path = os.path.abspath(log_path)
        self.host = host
        self.port = int(port)
        self.cores = cores
        self.run_as = run_as
        self.python = python or sys.executable
        self.stop_grace_s = stop_grace_s
        self.user_dir = os.path.join(self.work_dir, "engine")
        self.venv = os.path.join(self.user_dir, "venv")
        self.home = os.path.join(self.user_dir, "home")
        self.proc = None
        self.t_launch = None
        self.deadline = None  # perf_counter time by which install/build must end (grading budget)
        self._log = None
        self._uid = self._gid = None
        self._check_tools()
        os.makedirs(self.home, exist_ok=True)
        if run_as:
            os.chmod(self.work_dir, 0o755)
            for p in (self.user_dir, self.home):
                os.chown(p, self._uid, self._gid)

    # ------------------------------------------------------------------ setup

    def _check_tools(self):
        if not shutil.which("uv"):
            raise LaunchError("config", "uv is not on PATH (needed for `uv venv` and `uv pip install`)")
        if self.cores and not shutil.which("taskset"):
            raise LaunchError("config", "--cores given but taskset is not available")
        if self.run_as:
            import pwd
            try:
                pw = pwd.getpwnam(self.run_as)
            except KeyError:
                raise LaunchError("config", "--run-as user %r does not exist" % self.run_as)
            self._uid, self._gid = pw.pw_uid, pw.pw_gid
            if os.geteuid() != 0:
                raise LaunchError("config", "--run-as needs the grader to run as root")
            if not (shutil.which("setpriv") or shutil.which("sudo")):
                raise LaunchError("config", "--run-as needs setpriv or sudo")

    def env(self):
        env = engine_base_env()
        env.update({
            "VIRTUAL_ENV": self.venv,
            "PATH": os.path.join(self.venv, "bin") + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self.home,
            "MODEL_DIR": self.model_dir,
            "HOST": self.host,
            "PORT": str(self.port),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        })
        env.setdefault("UV_CACHE_DIR", os.path.join(self.home, ".cache", "uv"))
        return env

    def wrap(self, argv, pin=False, env=None):
        prefix = []
        if self.run_as:
            if shutil.which("setpriv"):
                prefix += ["setpriv", "--reuid=%d" % self._uid, "--regid=%d" % self._gid, "--init-groups", "--"]
            else:
                # sudo may reset the environment, so pass it explicitly through env(1).
                prefix += ["sudo", "-n", "-u", self.run_as, "--", "env"]
                prefix += ["%s=%s" % kv for kv in sorted((env or self.env()).items())]
        if pin and self.cores:
            prefix += ["taskset", "-c", str(self.cores)]
        return prefix + list(argv)

    def _open_log(self):
        if self._log is None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log = open(self.log_path, "ab", buffering=0)
        return self._log

    def log_line(self, text):
        self._open_log().write(("\n===== [grader] %s\n" % text).encode("utf-8"))

    def _run(self, stage, argv, timeout_s, capture=False):
        """Run a command in its own process group; kill the whole group on timeout."""
        if self.deadline is not None:
            left = self.deadline - time.perf_counter()
            if left <= 0:
                raise LaunchError("time_budget", "the grading time budget ran out before %s" % stage)
            if left < timeout_s:
                timeout_s = int(left)
        env = self.env()
        argv = self.wrap(argv, env=env)
        log = self._open_log()
        p = subprocess.Popen(argv, cwd=self.engine_dir, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE if capture else log, stderr=log, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise LaunchError(stage, "%s exceeded the %d s limit" % (stage, timeout_s))
        finally:
            _killpg(p.pid, signal.SIGKILL)  # anything the step left running in its group
        if p.returncode != 0:
            raise LaunchError(stage, "%s exited with code %d" % (stage, p.returncode))
        return out

    def create_venv(self):
        self.log_line("uv venv --system-site-packages")
        self._run("install", ["uv", "venv", "--system-site-packages", "--python", self.python, self.venv],
                  VENV_TIMEOUT_S)

    def install_requirements(self):
        req = os.path.join(self.engine_dir, "requirements.txt")
        if not os.path.isfile(req):
            return False
        self.log_line("uv pip install -r requirements.txt")
        self._run("install", ["uv", "pip", "install", "--python", os.path.join(self.venv, "bin", "python"),
                              "-r", req], INSTALL_TIMEOUT_S)
        return True

    def run_build(self):
        if not os.path.isfile(os.path.join(self.engine_dir, "build.sh")):
            return False
        self.log_line("bash build.sh")
        self._run("build", ["bash", "build.sh"], BUILD_TIMEOUT_S)
        return True

    def list_installed(self):
        """Distributions visible to the engine's Python, run as the engine user."""
        out = self._run("scan", [os.path.join(self.venv, "bin", "python"), "-c", LIST_DISTS_SCRIPT], 120,
                        capture=True)
        for line in reversed(out.decode("utf-8", "replace").splitlines()):
            if line.startswith("ARENA-DISTS "):
                return json.loads(line[len("ARENA-DISTS "):])["dists"]
        raise LaunchError("scan", "could not list the installed packages of the engine environment")

    def local_roots(self):
        return [self.venv, self.home, self.engine_dir]

    # ------------------------------------------------------------------ run

    def start(self):
        if port_open(self.host, self.port):
            raise LaunchError("launch", "something is already listening on %s:%d" % (self.host, self.port))
        if not os.path.isfile(os.path.join(self.engine_dir, "run.sh")):
            raise LaunchError("launch", "run.sh not found in the engine directory")
        env = self.env()
        argv = self.wrap(["bash", "run.sh"], pin=True, env=env)
        self.log_line("start: %s" % " ".join(a for a in argv if "=" not in a or a.startswith("--")))
        log = self._open_log()
        self.t_launch = time.perf_counter()
        self.proc = subprocess.Popen(argv, cwd=self.engine_dir, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        return self.t_launch

    def exit_code(self):
        return None if self.proc is None else self.proc.poll()

    def _group_alive(self):
        try:
            os.killpg(self.proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def stop(self):
        """SIGTERM the process group, SIGKILL after the grace period. Returns what happened."""
        info = {"started": self.proc is not None}
        if self.proc is None:
            return info
        t0 = time.perf_counter()
        self.proc.poll()
        if self.proc.returncode is not None and not self._group_alive():
            info.update(already_exited=True, exit_code=self.proc.returncode)
        else:
            _killpg(self.proc.pid, signal.SIGTERM)
            deadline = t0 + self.stop_grace_s
            while time.perf_counter() < deadline:
                self.proc.poll()
                if self.proc.returncode is not None and not self._group_alive():
                    break
                time.sleep(0.05)
            self.proc.poll()
            exited = self.proc.returncode is not None and not self._group_alive()
            info["sigterm_exit_s"] = round(time.perf_counter() - t0, 3) if exited else None
            info["killed"] = not exited
            if not exited:
                _killpg(self.proc.pid, signal.SIGKILL)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            info["exit_code"] = self.proc.returncode
        if self.run_as and shutil.which("pkill"):
            r = subprocess.run(["pkill", "-KILL", "-u", str(self._uid)], capture_output=True)
            info["killed_leftover_user_processes"] = r.returncode == 0
        deadline = time.perf_counter() + 5
        while port_open(self.host, self.port) and time.perf_counter() < deadline:
            time.sleep(0.1)
        info["port_closed"] = not port_open(self.host, self.port)
        if self._log is not None:
            self._log.close()
            self._log = None
        return info


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def tail_lines(path, n=60, max_line=500):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").splitlines()[-n:]
    return [ln[:max_line] for ln in lines]

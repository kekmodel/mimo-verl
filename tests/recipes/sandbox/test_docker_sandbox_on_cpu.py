# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Docker sandbox backend against a fake Engine API on a unix socket (no daemon needed),
plus one test against a real daemon when one is reachable."""

import base64
import io
import json
import os
import shutil
import socketserver
import struct
import tarfile
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipes.sandbox import docker_api
from recipes.sandbox.docker_env import DockerSandboxEnvironment, _node_slot, parse_bytes, parse_cpus

REPO_ROOT = Path(__file__).parents[3]


def _frame(stream: int, data: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + struct.pack(">I", len(data)) + data


class FakeDaemon:
    """Just enough of the Engine API for one sandbox at a time."""

    def __init__(self):
        # AF_UNIX paths are capped near 100 bytes; pytest's tmp_path can be longer.
        self._dir = tempfile.mkdtemp(prefix="fd", dir="/tmp")
        self.sock = os.path.join(self._dir, "docker.sock")
        self.images = {"img:1"}
        self.containers: dict[str, dict] = {}
        self.execs: dict[str, dict] = {}
        self.files: dict[str, bytes] = {}  # absolute path -> content (files only)
        self.calls: list[tuple[str, str, dict]] = []
        self.pull_error = None
        self.exec_script = None  # callable(argv) -> (frames, exit_code, delay)
        daemon = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, status, body=b"", ctype="application/json", close=False):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                if close:
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                    self.close_connection = True
                else:
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def _route(self, method):
                url = urllib.parse.urlsplit(self.path)
                query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                path = url.path.split("/", 2)[2] if url.path.startswith("/v") else url.path.lstrip("/")
                daemon.calls.append((method, "/" + path, {"query": query, "headers": dict(self.headers), "raw": raw}))
                return getattr(daemon, "h_" + method.lower())(self, "/" + path, query, raw)

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

            def do_PUT(self):
                self._route("PUT")

            def do_DELETE(self):
                self._route("DELETE")

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

            def get_request(self):
                request, _ = super().get_request()
                return request, ("local", 0)

        self.server = Server(self.sock, Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self._dir, ignore_errors=True)

    # -- handlers ---------------------------------------------------------------------------
    def h_get(self, h, path, query, raw):
        if path == "/version":
            return h._send(200, {"Version": "20.10.17", "ApiVersion": "1.41"})
        parts = path.strip("/").split("/")
        if parts[0] == "containers" and parts[-1] == "json" and len(parts) == 3:
            return h._send(200 if parts[1] in self.containers else 404, {"message": "no such container"})
        if parts[0] == "exec" and parts[-1] == "json":
            e = self.execs[parts[1]]
            return h._send(200, {"Running": False, "ExitCode": e.get("rc")})
        if parts[0] == "containers" and parts[-1] == "archive":
            src = query["path"]
            hits = {p: c for p, c in self.files.items() if p == src or p.startswith(src + "/")}
            if not hits:
                return h._send(404, {"message": "not found"})
            buf = io.BytesIO()
            base = os.path.dirname(src)
            with tarfile.open(fileobj=buf, mode="w") as tf:
                for p, c in hits.items():
                    info = tarfile.TarInfo(os.path.relpath(p, base))
                    info.size = len(c)
                    tf.addfile(info, io.BytesIO(c))
            return h._send(200, buf.getvalue(), "application/x-tar")
        return h._send(404, {"message": f"unhandled GET {path}"})

    def h_post(self, h, path, query, raw):
        body = json.loads(raw) if raw else {}
        parts = path.strip("/").split("/")
        if path == "/containers/create":
            if body["Image"] not in self.images:
                return h._send(404, {"message": f"No such image: {body['Image']}"})
            cid = f"c{len(self.containers)}" + "0" * 20
            self.containers[cid] = {"body": body, "name": query.get("name")}
            return h._send(201, {"Id": cid, "Warnings": []})
        if path == "/images/create":
            if self.pull_error:
                return h._send(
                    200, b'{"status":"Pulling"}\n{"error":"' + self.pull_error.encode() + b'"}\n', close=True
                )
            self.images.add(f"{query['fromImage']}:{query['tag']}")
            return h._send(200, b'{"status":"Pulling"}\n{"status":"Downloaded"}\n', close=True)
        if parts[0] == "containers" and parts[-1] == "start":
            return h._send(204)
        if parts[0] == "containers" and parts[-1] == "exec":
            if parts[1] not in self.containers:
                return h._send(409, {"message": "container is not running"})
            eid = f"e{len(self.execs)}"
            self.execs[eid] = {"cmd": body["Cmd"]}
            return h._send(201, {"Id": eid})
        if parts[0] == "exec" and parts[-1] == "start":
            e = self.execs[parts[1]]
            frames, rc, delay = self.exec_script(e["cmd"]) if self.exec_script else ([], 0, 0)
            e["rc"] = rc
            h.send_response(200)
            h.send_header("Content-Type", "application/vnd.docker.raw-stream")
            h.send_header("Connection", "close")
            h.end_headers()
            for f in frames:
                h.wfile.write(f)
                h.wfile.flush()
            time.sleep(delay)
            h.close_connection = True
            return None
        return h._send(404, {"message": f"unhandled POST {path}"})

    def h_put(self, h, path, query, raw):
        parts = path.strip("/").split("/")
        assert parts[-1] == "archive" and query["path"] == "/"
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for m in tf.getmembers():
                assert m.uid == 0 and m.gid == 0
                if m.isfile():
                    self.files["/" + m.name] = tf.extractfile(m).read()
        return h._send(200, b"", "text/plain")

    def h_delete(self, h, path, query, raw):
        cid = path.strip("/").split("/")[1]
        existed = self.containers.pop(cid, None)
        return h._send(204 if existed else 404, b"" if existed else {"message": "gone"})


@pytest.fixture
def daemon(tmp_path):
    d = FakeDaemon()
    yield d
    d.close()


def _env(daemon, tmp_path, **kwargs):
    kw = dict(image="img:1", docker_host=f"unix://{daemon.sock}", lock_dir=str(tmp_path / "locks"), timeout=5)
    kw.update(kwargs)
    env = DockerSandboxEnvironment(**kw)
    env.instance_id = "repo__issue-1"
    return env


def test_create_body_is_a_capped_labelled_keepalive(daemon, tmp_path, monkeypatch):
    monkeypatch.setenv("EXP_NAME", "exp1")
    env = _env(daemon, tmp_path, cpu_limit="500m", memory_limit="2Gi", labels={"team": "rl"}, max_lifetime=600)
    env.start()
    created = next(c for c in daemon.containers.values())
    body = created["body"]
    assert body["Entrypoint"] == ["/bin/sh", "-c"] and body["Cmd"] == ["sleep 600"]  # image ENTRYPOINT replaced
    host = body["HostConfig"]
    assert host["NetworkMode"] == "none" and host["AutoRemove"] is True
    assert host["NanoCpus"] == 500_000_000 and host["Memory"] == 2 * 2**30 and host["PidsLimit"] == 4096
    assert host["CpuShares"] == 512 and host["MemoryReservation"] == 2**30  # cpu_request 0.5, memory_request 1Gi
    labels = body["Labels"]
    assert labels["mimo.sandbox"] == "1" and labels["mimo.exp"] == "exp1" and labels["team"] == "rl"
    assert labels["mimo.instance"] == "repo__issue-1" and created["name"].startswith("mimo-repo__issue-1-")
    env.cleanup()
    assert env.container_id is None and not daemon.containers
    env.cleanup()  # idempotent


def test_missing_image_is_pulled_with_registry_auth(daemon, tmp_path):
    auth = tmp_path / "config.json"
    auth.write_text(json.dumps({"auths": {"kcr.example": {"auth": base64.b64encode(b"jd:tok").decode()}}}))
    env = _env(daemon, tmp_path, image="kcr.example/grp/task:7", registry_auth_file=str(auth))
    env.start()
    pull = next(c for c in daemon.calls if c[1] == "/images/create")
    assert pull[2]["query"] == {"fromImage": "kcr.example/grp/task", "tag": "7"}
    sent = json.loads(base64.urlsafe_b64decode(pull[2]["headers"]["X-Registry-Auth"]))
    assert sent == {"username": "jd", "password": "tok", "serveraddress": "kcr.example"}
    assert env.container_id
    env.cleanup()


def test_pull_policy_never_and_pull_errors(daemon, tmp_path):
    with pytest.raises(docker_api.DockerAPIError):
        _env(daemon, tmp_path, image="absent:1", pull_policy="never").start()
    assert not any(c[1] == "/images/create" for c in daemon.calls)
    daemon.pull_error = "denied: access forbidden"
    with pytest.raises(docker_api.DockerAPIError, match="access forbidden"):
        _env(daemon, tmp_path, image="absent:1").start()
    with pytest.raises(ValueError):
        _env(daemon, tmp_path, pull_policy="sometimes")


def test_execute_follows_the_kubernetes_contract(daemon, tmp_path):
    env = _env(daemon, tmp_path, cwd="/testbed")
    env.start()
    daemon.exec_script = lambda argv: ([_frame(1, b"out\n"), _frame(2, b"err\n")], 3, 0)
    res = env.execute("make test", timeout=7)
    assert res == {"output": "out\nerr\n", "returncode": 3, "reason": "ok"}
    argv = daemon.execs["e0"]["cmd"]
    assert argv == ["timeout", "7", "/bin/bash", "-lc", "cd /testbed && make test"]

    env.execute("ls", cwd="/other dir", as_user="bob")
    assert daemon.execs["e1"]["cmd"][4] == "su bob -s /bin/sh -c 'cd '\"'\"'/other dir'\"'\"' && ls'"

    daemon.exec_script = lambda argv: ([_frame(1, b"partial")], 124, 0)
    res = env.execute("sleep 99", timeout=1)
    assert res["reason"] == "pod_timeout" and res["returncode"] == 124 and res["output"].startswith("partial")

    # A frame split across reads is reassembled.
    payload = b"x" * 70000
    daemon.exec_script = lambda argv: ([_frame(1, payload)[:9], _frame(1, payload)[9:]], 0, 0)
    assert env.execute("cat big")["output"] == payload.decode()
    env.cleanup()


def test_client_deadline_and_lost_sandbox(daemon, tmp_path, monkeypatch):
    import recipes.sandbox.docker_env as de

    monkeypatch.setattr(de, "_EXEC_GRACE_S", 0)
    env = _env(daemon, tmp_path)
    env.start()
    daemon.exec_script = lambda argv: ([_frame(1, b"started\n")], 0, 3)  # never finishes in time
    t0 = time.monotonic()
    res = env.execute("hang", timeout=1)
    assert res["reason"] == "client_timeout" and res["returncode"] is None and "started" in res["output"]
    assert time.monotonic() - t0 < 2.5

    daemon.containers.clear()  # sandbox died (OOM, killed)
    res = env.execute("echo hi")
    assert res["reason"] == "transport_error" and res["returncode"] is None

    class TransportError(RuntimeError):
        pass

    monkeypatch.setitem(
        __import__("sys").modules, "mimoagent.environments", SimpleNamespace(TransportError=TransportError)
    )
    env.config.raise_on_transport_error = True
    with pytest.raises(TransportError):
        env.execute("echo hi")


def test_copy_to_and_copy_out_place_the_entry_at_the_destination(daemon, tmp_path):
    env = _env(daemon, tmp_path, cwd="/testbed")
    env.start()
    src = tmp_path / "patch.diff"
    src.write_text("diff\n")
    env.copy_to(str(src), "/run/tests/test.patch")  # missing parents: unpacked at "/"
    env.copy_to(str(src), "relative.txt")  # relative to cwd
    env.copy_to(str(src), "/into/dir/")  # trailing slash keeps the source name
    tree = tmp_path / "tree" / "sub"
    tree.mkdir(parents=True)
    (tree / "x.py").write_text("x = 1\n")
    env.copy_to(str(tmp_path / "tree"), "/testbed/pkg")
    assert daemon.files == {
        "/run/tests/test.patch": b"diff\n",
        "/testbed/relative.txt": b"diff\n",
        "/into/dir/patch.diff": b"diff\n",
        "/testbed/pkg/sub/x.py": b"x = 1\n",
    }
    put = [c for c in daemon.calls if c[0] == "PUT"]
    assert all(c[2]["query"] == {"path": "/"} for c in put)

    out = tmp_path / "out"
    env.copy_out("/testbed/pkg", str(out / "copied"))
    assert (out / "copied" / "sub" / "x.py").read_text() == "x = 1\n"
    env.copy_out("/run/tests/test.patch", str(out / "p.diff"))
    assert (out / "p.diff").read_text() == "diff\n"
    with pytest.raises(FileNotFoundError):
        env.copy_out("/nope", str(out / "n"))
    with pytest.raises(FileNotFoundError):
        env.copy_to(str(tmp_path / "absent"), "/x")
    env.cleanup()


def test_node_slots_bound_concurrency_across_holders(tmp_path):
    lock_dir = str(tmp_path / "locks")
    held = threading.Event()
    release = threading.Event()

    def holder():
        with _node_slot(lock_dir, "start", 1, 5):
            held.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    held.wait(5)
    # flock is per open file description, so a second holder in this process also waits.
    with pytest.raises(TimeoutError):
        with _node_slot(lock_dir, "start", 1, 0.3):
            pass
    release.set()
    t.join()
    with _node_slot(lock_dir, "start", 1, 1):
        pass
    with _node_slot(lock_dir, "start", 0, 0):  # 0 = unlimited
        pass


def test_quantities_and_image_names():
    assert parse_bytes("8Gi") == 8 * 2**30 and parse_bytes("512Mi") == 512 * 2**20 and parse_bytes("1G") == 10**9
    assert parse_bytes(0) == 0 and parse_bytes("1048576") == 2**20
    assert parse_cpus("4") == 4 and parse_cpus("250m") == 0.25 and parse_cpus("0") == 0
    with pytest.raises(ValueError):
        parse_bytes("lots")
    assert docker_api.split_image("kcr.x:5000/a/b:tag") == ("kcr.x:5000/a/b", "tag")
    assert docker_api.split_image("kcr.x:5000/a/b") == ("kcr.x:5000/a/b", "latest")
    assert docker_api.split_image("format-code-task-1:latest") == ("format-code-task-1", "latest")


def test_null_environment_override_removes_the_harness_key(monkeypatch):
    from recipes.code import mimoagent_runner as runner

    seen = {}

    class Stop(Exception):
        pass

    def fake_make(instance, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", fake_make)
    with pytest.raises(Stop):
        runner._run_sync(
            raw_prompt="x",
            instance={},
            session=SimpleNamespace(base_url="http://gateway/s/v1"),
            config={"environment": {"environment_class": "kubernetes", "node_selector": {}, "cwd": "/testbed"}},
            agent_overrides={},
            environment_overrides={
                "environment_class": "recipes.sandbox.docker_env.DockerSandboxEnvironment",
                "node_selector": None,
            },
        )
    assert seen == {"environment_class": "recipes.sandbox.docker_env.DockerSandboxEnvironment", "cwd": "/testbed"}


def test_sandbox_docker_config_group_composes(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "kcr.example/grp/mimo-tasks")
    with initialize_config_dir(config_dir=str(REPO_ROOT / "recipes/code/config"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"hydra.searchpath=[pkg://verl.trainer.config,file://{REPO_ROOT}/recipes/code/config]",
                "+sandbox=docker",
            ],
        )
    OmegaConf.resolve(cfg)
    kwargs = cfg.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mimoagent.runner_kwargs
    overrides = OmegaConf.to_container(kwargs.environment_overrides)
    assert overrides["environment_class"] == "recipes.sandbox.docker_env.DockerSandboxEnvironment"
    assert overrides["image_prefix"] == "kcr.example/grp/mimo-tasks"
    assert overrides["node_selector"] is None and overrides["docker_host"] is None
    assert overrides["network_mode"] == "none" and overrides["pull_policy"] == "missing"
    assert overrides["max_concurrent_starts"] == 8 and overrides["max_lifetime"] == 14400
    assert kwargs.exec_budget_seconds  # the rest of runner_kwargs is untouched
    # Every key the group sets is a backend config field or one make_dataset_env consumes.
    from dataclasses import fields

    from recipes.sandbox.docker_env import DockerSandboxConfig

    known = {f.name for f in fields(DockerSandboxConfig)} | {"environment_class", "image_prefix", "node_selector"}
    assert set(overrides) <= known


def _real_daemon_image():
    image = os.environ.get("SANDBOX_IT_IMAGE", "python:3.13-slim")
    try:
        client = docker_api.DockerClient()
        client.version()
        client.request("GET", f"/images/{image}/json", timeout=5)
    except Exception:
        return None
    return image


@pytest.mark.skipif(_real_daemon_image() is None, reason="no reachable Docker daemon with the test image")
def test_against_a_real_daemon(tmp_path):
    env = DockerSandboxEnvironment(
        image=_real_daemon_image(), cwd="/tmp", timeout=20, cpu_limit="1", memory_limit="512Mi",
        max_lifetime=300, lock_dir=str(tmp_path / "locks"),
    )  # fmt: skip
    env.instance_id = "it"
    env.start()
    try:
        res = env.execute("echo out; echo err >&2; exit 4")
        # stdout and stderr are separate pipes: their relative order is not guaranteed.
        assert (sorted(res["output"].splitlines()), res["returncode"], res["reason"]) == (["err", "out"], 4, "ok")
        assert env.execute("sleep 5", timeout=1)["reason"] == "pod_timeout"
        src = tmp_path / "f.txt"
        src.write_text("hello")
        env.copy_to(str(src), "/deep/a/b.txt")
        assert env.execute("cat /deep/a/b.txt")["output"] == "hello"
        env.copy_out("/deep/a", str(tmp_path / "back"))
        assert (tmp_path / "back" / "b.txt").read_text() == "hello"
        assert env.execute("cat /proc/net/route | wc -l")["output"].strip() == "1"  # network none: header only
    finally:
        cid = env.container_id
        env.cleanup()
    with pytest.raises(docker_api.DockerAPIError):
        docker_api.DockerClient().request("GET", f"/containers/{cid}/json", timeout=5)

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
"""Task sandboxes as sibling containers on the node's Docker daemon (Docker-outside-of-Docker).

For clusters without Kubernetes: the training container mounts the node's
``/var/run/docker.sock`` and every rollout gets its own container next to it. Selected with
``environment_class: recipes.sandbox.docker_env.DockerSandboxEnvironment`` (``SANDBOX=docker``
in ``recipes/code/run_train.sh``).

The execution contract is mimoagent's Kubernetes backend, which the tools and the runner are
written against: commands run as ``timeout N /bin/bash -lc`` inside the sandbox (rc 124 on
expiry), results are ``{"output", "returncode", "reason"}`` with ``reason`` in ``ok``,
``pod_timeout``, ``client_timeout``, ``transport_error``, ``raise_on_transport_error`` turns a
lost sandbox into ``TransportError``, and ``copy_to`` / ``copy_out`` place the entry at
``dest_path``.

Unlike mimoagent's ``docker`` backend it needs no ``docker`` CLI (the API is called over the
socket), uses no bind mounts (files move as tar archives, which also passes daemons whose
authorization plugin forbids mounts), caps resources, labels every sandbox, bounds its
lifetime so an orphan dies on its own, and removes it synchronously.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import posixpath
import random
import re
import shlex
import shutil
import socket
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from .docker_api import DockerAPIError, DockerClient, DockerTransportError, demux, registry_auth_header

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 50 * 1024 * 1024
_EXEC_GRACE_S = 5
_SPOOL_BYTES = 32 << 20


class _ImageMissing(DockerAPIError):
    """Create answered 404: the node does not have the image."""


@dataclass
class DockerSandboxConfig:
    image: str
    cwd: str = "/testbed"
    """Working directory commands run in."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables set in the sandbox."""
    forward_env: list[str] = field(default_factory=list)
    """Variables copied from the runner's environment when set there (``env`` wins)."""
    timeout: int = 30
    """Default per-command timeout (seconds)."""
    raise_on_transport_error: bool = False
    """Raise TransportError when the sandbox cannot be reached instead of returning
    ``reason=transport_error``."""

    # Resources: the Kubernetes backend's names, so the harness yamls carry over.
    cpu_limit: str = "4"
    """CPU cap (``"4"``, ``"500m"``); 0 = none."""
    memory_limit: str = "8Gi"
    """Memory cap (``"8Gi"``, ``"512Mi"``); 0 = none."""
    cpu_request: str = "0.5"
    """Relative CPU weight under contention (Docker ``CpuShares``: 1024 per CPU)."""
    memory_request: str = "1Gi"
    """Soft memory reservation (Docker ``MemoryReservation``)."""
    pids_limit: int = 4096
    """Process cap (fork-bomb guard); 0 = none."""
    host_network: bool = False
    """Use the node's network (overrides ``network_mode``)."""
    network_mode: str = "none"
    """Docker network for the sandbox. ``none``: no network, which code tasks with prebuilt
    images do not need and which keeps the agent off the node's network."""

    labels: dict[str, str] = field(default_factory=dict)
    """Extra labels; every sandbox also gets ``mimo.sandbox=1``, ``mimo.exp``,
    ``mimo.instance`` and ``mimo.owner`` (the runner's hostname)."""
    max_lifetime: int = 14400
    """Seconds after which the sandbox exits by itself (and is removed with
    ``auto_remove``), so a sandbox orphaned by a dead trainer does not live on. Keep it above
    the session backstop plus grading."""
    auto_remove: bool = True
    """Remove the container when it exits (Docker ``AutoRemove``)."""
    init: bool = False
    """Run Docker's init as PID 1. Without it nothing reaps the orphans of commands killed by
    ``timeout``, and their zombies count against ``pids_limit``."""

    docker_host: str | None = None
    """``unix:///var/run/docker.sock`` (default, or ``$DOCKER_HOST``) or ``tcp://host:port``."""
    api_version: str = "1.41"
    """Engine API version in request paths (1.41 = Docker 20.10)."""
    pull_policy: str = "missing"
    """``missing``: pull when the node lacks the image; ``always``; ``never`` (pre-pulled
    nodes; a missing image fails the sandbox)."""
    pull_timeout: int = 1800
    """Seconds one pull may take (task images run to several GB)."""
    registry_auth_file: str | None = None
    """Docker ``config.json``-style credentials for pulls (default
    ``$MIMO_REGISTRY_AUTH_FILE``). The daemon does not use the node's ``docker login`` for
    API pulls. Never put the credentials themselves in the config."""
    start_timeout: int = 120
    """Seconds for creating and starting the container (pulls excluded)."""
    max_concurrent_starts: int = 8
    """Container creations at once per node (all runner processes sharing ``lock_dir``); the
    daemon is shared and stalls under bursts. 0 = unlimited."""
    max_concurrent_pulls: int = 2
    """Image pulls at once per node. 0 = unlimited."""
    lock_dir: str = "/tmp/mimo-sandbox-locks"
    """Directory of the per-node slot locks; must be local to the node."""


class DockerSandboxEnvironment:
    def __init__(self, *, config_class: type = DockerSandboxConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.pull_policy not in ("missing", "always", "never"):
            raise ValueError(f"pull_policy must be missing, always or never; got {self.config.pull_policy!r}")
        self.client = DockerClient(self.config.docker_host, self.config.api_version)
        self.container_id: str | None = None
        self.container_name: str | None = None
        self.instance_id: str | None = None
        self.logger = logger

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        if self.config.pull_policy == "always":
            self._pull()
        name = self._container_name()
        body = self._create_body()
        try:
            self._create_and_start(name, body)
        except _ImageMissing:
            if self.config.pull_policy == "never":
                raise
            self._pull()
            self._create_and_start(name, body)
        self.logger.info("sandbox %s started (image=%s instance=%s)", name, self.config.image, self.instance_id)

    def _create_and_start(self, name: str, body: dict[str, Any]) -> None:
        cfg = self.config
        with _node_slot(cfg.lock_dir, "start", cfg.max_concurrent_starts, cfg.start_timeout):
            try:
                created = self.client.request(
                    "POST", "/containers/create", query={"name": name}, body=body, ok=(201,), timeout=cfg.start_timeout
                )
            except DockerAPIError as e:
                if e.status == 404:
                    raise _ImageMissing(e.status, f"{cfg.image} is not on this node: {e.message}") from e
                raise
            self.container_id, self.container_name = created["Id"], name
            try:
                self.client.request(
                    "POST", f"/containers/{self.container_id}/start", ok=(204, 304), timeout=cfg.start_timeout
                )
            except Exception:
                self.cleanup(wait=False)
                raise

    def _pull(self) -> None:
        auth_file = self.config.registry_auth_file or os.environ.get("MIMO_REGISTRY_AUTH_FILE")
        auth = registry_auth_header(self.config.image, auth_file)
        t0 = time.monotonic()
        with _node_slot(self.config.lock_dir, "pull", self.config.max_concurrent_pulls, self.config.pull_timeout):
            self.client.pull(self.config.image, auth=auth, timeout=self.config.pull_timeout)
        self.logger.info("pulled %s in %.0fs", self.config.image, time.monotonic() - t0)

    def _container_name(self) -> str:
        stem = re.sub(r"[^a-zA-Z0-9_.-]", "-", self.instance_id or "sandbox")[:40].strip("-_.") or "sandbox"
        return f"mimo-{stem}-{uuid.uuid4().hex[:8]}"

    def _create_body(self) -> dict[str, Any]:
        cfg = self.config
        env = {k: v for k in cfg.forward_env if (v := os.environ.get(k)) is not None}
        env.update({k: str(v) for k, v in cfg.env.items()})
        labels = {
            "mimo.sandbox": "1",
            "mimo.exp": os.environ.get("EXP_NAME", "unknown"),
            "mimo.instance": self.instance_id or "",
            "mimo.owner": socket.gethostname(),
            **{str(k): str(v) for k, v in (cfg.labels or {}).items()},
        }
        host: dict[str, Any] = {
            "NetworkMode": "host" if cfg.host_network else cfg.network_mode,
            "AutoRemove": bool(cfg.auto_remove),
        }
        if (cpus := parse_cpus(cfg.cpu_limit)) > 0:
            host["NanoCpus"] = int(cpus * 1e9)
        if (memory := parse_bytes(cfg.memory_limit)) > 0:
            host["Memory"] = memory
            host["MemorySwap"] = memory  # no swap on top: a memory hog is OOM-killed, as on Kubernetes
        if (shares := parse_cpus(cfg.cpu_request)) > 0:
            host["CpuShares"] = max(2, int(shares * 1024))
        if (reservation := parse_bytes(cfg.memory_request)) > 0:
            # The daemon rejects a reservation above the limit.
            host["MemoryReservation"] = min(reservation, memory) if memory > 0 else reservation
        if cfg.pids_limit > 0:
            host["PidsLimit"] = int(cfg.pids_limit)
        if cfg.init:
            host["Init"] = True
        # The image's own ENTRYPOINT would turn the keep-alive into its arguments; replace it.
        # A plain integer sleep works with GNU and BusyBox alike, and ending it ends the sandbox.
        return {
            "Image": cfg.image,
            "Entrypoint": ["/bin/sh", "-c"],
            "Cmd": [f"sleep {int(cfg.max_lifetime)}"],
            "Env": [f"{k}={v}" for k, v in env.items()],
            "Labels": labels,
            "Tty": False,
            "OpenStdin": False,
            "HostConfig": host,
        }

    def cleanup(self, wait: bool = True) -> None:
        """Remove the container and wait until it is gone. Idempotent. ``wait=False`` sends one
        removal and returns (garbage collection, failed starts); ``max_lifetime`` backs it up."""
        container = getattr(self, "container_id", None)
        if not container:
            return
        if not wait:
            self.container_id = None
            try:
                self.client.request(
                    "DELETE", f"/containers/{container}", query={"force": 1, "v": 1}, ok=(204, 404, 409), timeout=10
                )
            except (DockerAPIError, DockerTransportError) as e:
                self.logger.warning("removing sandbox %s: %s", container[:12], e)
            return
        for attempt in range(5):
            try:
                self.client.request(
                    "DELETE", f"/containers/{container}", query={"force": 1, "v": 1}, ok=(204, 404), timeout=60
                )
                if self._gone(container):
                    break
            except DockerAPIError as e:
                if e.status != 409:  # 409: removal already in progress
                    self.logger.warning("removing sandbox %s: %s", container[:12], e)
            except DockerTransportError as e:
                self.logger.warning("removing sandbox %s: %s", container[:12], e)
            time.sleep(min(2**attempt, 10))
        else:
            self.logger.error("sandbox %s still present after cleanup; max_lifetime will end it", container[:12])
        self.container_id = None

    def _gone(self, container: str, wait: float = 15) -> bool:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                self.client.request("GET", f"/containers/{container}/json", timeout=10)
            except DockerAPIError as e:
                if e.status == 404:
                    return True
            time.sleep(0.5)
        return False

    def __del__(self):
        try:
            self.cleanup(wait=False)
        except Exception:  # noqa: BLE001 - interpreter teardown; max_lifetime is the backstop
            pass

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    # -- commands ----------------------------------------------------------------------------

    def _transport_error(self, output: str) -> dict[str, Any]:
        if self.config.raise_on_transport_error:
            from mimoagent.environments import TransportError

            raise TransportError(output)
        return {"output": output, "returncode": None, "reason": "transport_error"}

    def execute(
        self, command: str, cwd: str = "", timeout: int | None = None, *, as_user: str | None = None
    ) -> dict[str, Any]:
        """Run ``command`` in the sandbox; see the module docstring for the result contract."""
        assert self.container_id, "Sandbox not started"
        if timeout is None:
            timeout = self.config.timeout
        full = f"cd {shlex.quote(cwd or self.config.cwd)} && {command}"
        if as_user:
            full = f"su {shlex.quote(as_user)} -s /bin/sh -c {shlex.quote(full)}"
        argv = ["timeout", str(timeout), "/bin/bash", "-lc", full]
        deadline = time.monotonic() + timeout + _EXEC_GRACE_S
        try:
            exec_id = self.client.request(
                "POST",
                f"/containers/{self.container_id}/exec",
                body={"Cmd": argv, "AttachStdout": True, "AttachStderr": True, "Tty": False},
                ok=(201,),
                timeout=30,
            )["Id"]
        except (DockerAPIError, DockerTransportError) as e:
            return self._transport_error(f"Error creating exec: {e}")

        try:
            output, timed_out = self._read_exec(exec_id, deadline)
        except DockerTransportError as e:
            return self._transport_error(f"Exec stream failed: {e}")
        if timed_out:
            return {
                "output": output + f"\nCommand timed out (client) after {timeout}s",
                "returncode": None,
                "reason": "client_timeout",
            }

        rc = self._exit_code(exec_id)
        if rc is None:
            return self._transport_error(output or "Exec finished without an exit code")
        if rc == 124:
            return {
                "output": output + f"\nCommand timed out (pod) after {timeout}s",
                "returncode": 124,
                "reason": "pod_timeout",
            }
        return {"output": output, "returncode": rc, "reason": "ok"}

    def _read_exec(self, exec_id: str, deadline: float) -> tuple[str, bool]:
        """Start the exec attached and collect stdout+stderr in arrival order until it ends or
        ``deadline`` passes. Keeps the last 50 MB."""
        remaining = max(1.0, deadline - time.monotonic())
        conn, resp = self.client.open(
            "POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, timeout=remaining
        )
        buf = bytearray()
        out = bytearray()
        timed_out = False
        try:
            if resp.status != 200:
                raise DockerTransportError(f"exec start -> {resp.status}: {resp.read()[:300]!r}")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                if conn.sock is not None:
                    conn.sock.settimeout(remaining)
                try:
                    chunk = resp.read1(65536)
                except TimeoutError:
                    timed_out = True
                    break
                except OSError as e:
                    raise DockerTransportError(repr(e)) from e
                if not chunk:
                    break
                buf += chunk
                for _stream, payload in demux(buf):
                    out += payload
                if len(out) > _MAX_OUTPUT_BYTES:
                    del out[: len(out) - _MAX_OUTPUT_BYTES]
        finally:
            conn.close()
        return out.decode("utf-8", "replace"), timed_out

    def _exit_code(self, exec_id: str) -> int | None:
        for _ in range(20):
            try:
                info = self.client.request("GET", f"/exec/{exec_id}/json", timeout=10)
            except (DockerAPIError, DockerTransportError):
                return None
            if not info.get("Running"):
                return info.get("ExitCode")
            time.sleep(0.1)
        return None

    def execute_detached(self, command: str, cwd: str = "", timeout: int | None = None, **_ignored) -> dict[str, Any]:
        """A local socket has no connection lifetime cap: a plain execute."""
        return self.execute(command, cwd=cwd, timeout=timeout)

    # -- files -------------------------------------------------------------------------------

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else posixpath.join(self.config.cwd, path)

    def copy_to(
        self,
        src_path: str,
        dest_path: str,
        *,
        timeout: int = 300,
        max_retries: int = 3,
        as_user: str | None = None,
        dereference: bool = False,
    ) -> None:
        """Copy a local file or directory to ``dest_path`` in the sandbox (the entry is renamed
        to ``dest_path``'s basename; missing parent directories are created)."""
        assert self.container_id, "Sandbox not started"
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path not found: {src_path}")
        dest = self._abs(dest_path).rstrip("/") or "/"
        if dest_path.endswith("/"):
            dest = posixpath.join(dest, os.path.basename(os.path.normpath(src_path)))
        now = time.time()

        def as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = now
            return info

        last: Exception | None = None
        for attempt in range(1, max_retries + 1):
            with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as archive:
                # Entries carry the full destination path and are unpacked at "/", so the
                # daemon creates missing parent directories.
                with tarfile.open(fileobj=archive, mode="w", dereference=dereference) as tf:
                    tf.add(src_path, arcname=dest.lstrip("/"), recursive=True, filter=as_root)
                size = archive.tell()
                archive.seek(0)
                try:
                    self.client.request(
                        "PUT",
                        f"/containers/{self.container_id}/archive",
                        query={"path": "/"},
                        body=archive,
                        headers={"Content-Type": "application/x-tar", "Content-Length": str(size)},
                        ok=(200,),
                        timeout=timeout,
                    )
                    last = None
                    break
                except (DockerAPIError, DockerTransportError) as e:
                    last = e
                    if isinstance(e, DockerAPIError) and e.status in (400, 403, 404):
                        break  # not transient: bad path, refused, sandbox gone
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
        if last is not None:
            raise RuntimeError(f"copy_to {dest} failed: {last}")
        if as_user:
            res = self.execute(f"chown -R {shlex.quote(as_user)} {shlex.quote(dest)}", timeout=60)
            if res.get("returncode") != 0:
                raise RuntimeError(f"chown {dest} to {as_user}: {res.get('output', '')[-300:]}")

    def copy_out(
        self, src_path: str, dest_path: str, *, timeout: int = 300, max_retries: int = 3, as_user: str | None = None
    ) -> None:
        """Copy ``src_path`` (file or directory) from the sandbox to local ``dest_path``."""
        assert self.container_id, "Sandbox not started"
        src = self._abs(src_path).rstrip("/") or "/"
        src_base = posixpath.basename(src)
        if not src_base:
            raise ValueError(f"invalid src_path (no basename): {src_path!r}")
        dest = os.path.abspath(dest_path)
        dest_parent = os.path.dirname(dest) or "."
        last: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as archive:
                    conn, resp = self.client.open(
                        "GET", f"/containers/{self.container_id}/archive", query={"path": src}, timeout=timeout
                    )
                    try:
                        if resp.status == 404:
                            raise FileNotFoundError(f"{src} not found in sandbox")
                        if resp.status != 200:
                            raise DockerAPIError(resp.status, resp.read()[:300].decode("utf-8", "replace"), "GET", src)
                        shutil.copyfileobj(resp, archive, 1 << 20)
                    finally:
                        conn.close()
                    archive.seek(0)
                    _extract_single(archive, src_base, dest, dest_parent)
                return
            except FileNotFoundError:
                raise
            except (DockerAPIError, DockerTransportError, OSError, tarfile.TarError) as e:
                last = e
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"copy_out {src} failed after {max_retries} attempts: {last}")


def _extract_single(archive, src_base: str, dest: str, dest_parent: str) -> None:
    os.makedirs(dest_parent, exist_ok=True)
    tmp_dir = os.path.join(dest_parent, f".copy_out_{uuid.uuid4().hex[:12]}")
    os.makedirs(tmp_dir)
    try:
        with tarfile.open(fileobj=archive, mode="r:") as tf:
            try:
                tf.extractall(tmp_dir, filter="tar")
            except TypeError:  # Python without extraction filters
                tf.extractall(tmp_dir)
        extracted = os.path.join(tmp_dir, src_base)
        if not os.path.lexists(extracted):
            entries = os.listdir(tmp_dir)
            if len(entries) != 1:
                raise RuntimeError(f"unexpected archive layout: {entries}")
            extracted = os.path.join(tmp_dir, entries[0])
        if os.path.lexists(dest):
            if os.path.isdir(dest) and not os.path.islink(dest):
                shutil.rmtree(dest)
            else:
                os.remove(dest)
        shutil.move(extracted, dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- helpers -----------------------------------------------------------------------------------


@contextmanager
def _node_slot(lock_dir: str, kind: str, slots: int, timeout: float):
    """Hold one of ``slots`` per-node slots (flock on files in ``lock_dir``, so every runner
    process on the node shares them and a dead holder frees its slot). 0 = no limit."""
    if slots <= 0:
        yield
        return
    os.makedirs(lock_dir, exist_ok=True)
    deadline = time.monotonic() + timeout
    order = list(range(slots))
    while True:
        random.shuffle(order)
        for i in order:
            fd = os.open(os.path.join(lock_dir, f"{kind}-{i}.lock"), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                if e.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
                continue
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"no free {kind} slot in {lock_dir} after {timeout:.0f}s")
        time.sleep(0.2 + random.random() * 0.3)


_BYTE_UNITS = {
    "": 1,
    "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12,
    "ki": 2**10, "mi": 2**20, "gi": 2**30, "ti": 2**40,
}  # fmt: skip


def parse_bytes(value: Any) -> int:
    """Kubernetes-style quantity (``8Gi``, ``512Mi``, ``1G``, ``1048576``) -> bytes."""
    if value in (None, ""):
        return 0
    m = re.fullmatch(r"\s*([0-9.]+)\s*([kKmMgGtT]i?)?[bB]?\s*", str(value))
    if not m:
        raise ValueError(f"bad memory quantity {value!r}")
    return int(float(m.group(1)) * _BYTE_UNITS[(m.group(2) or "").lower()])


def parse_cpus(value: Any) -> float:
    """``"4"``, ``"0.5"`` or millicores ``"500m"`` -> CPUs."""
    if value in (None, ""):
        return 0.0
    text = str(value).strip()
    if text.endswith("m"):
        return float(text[:-1]) / 1000
    return float(text)

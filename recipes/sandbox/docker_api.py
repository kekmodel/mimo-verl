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
"""A small Docker Engine API client over the daemon socket, standard library only.

The sandbox backend talks to the node's daemon through a mounted ``docker.sock`` (sibling
containers, "Docker-outside-of-Docker"). It does not need the ``docker`` CLI in the training
image or the ``docker`` Python package: only the handful of endpoints a sandbox uses.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import socket
import struct
import time
import urllib.parse
from typing import Any, Iterator

DEFAULT_SOCKET = "/var/run/docker.sock"


class DockerAPIError(RuntimeError):
    """The daemon answered with an error status."""

    def __init__(self, status: int, message: str, method: str = "", path: str = ""):
        super().__init__(f"{method} {path} -> {status}: {message}")
        self.status = status
        self.message = message


class DockerTransportError(RuntimeError):
    """The daemon could not be reached, or the connection broke mid-request."""


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float | None):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def resolve_docker_host(docker_host: str | None) -> str:
    """``unix:///path``, ``tcp://host:port`` or a bare socket path; default ``$DOCKER_HOST``,
    then ``/var/run/docker.sock``."""
    return docker_host or os.environ.get("DOCKER_HOST") or f"unix://{DEFAULT_SOCKET}"


def demux(buf: bytearray) -> Iterator[tuple[int, bytes]]:
    """Pop complete frames of a non-TTY attach stream (8-byte header: stream id, 3 zero
    bytes, big-endian payload size) off the front of ``buf``."""
    while len(buf) >= 8:
        stream_id = buf[0]
        (size,) = struct.unpack(">I", bytes(buf[4:8]))
        if len(buf) < 8 + size:
            return
        payload = bytes(buf[8 : 8 + size])
        del buf[: 8 + size]
        yield stream_id, payload


def registry_auth_header(image: str, auth_file: str | None) -> str | None:
    """``X-Registry-Auth`` for pulling ``image``, from a docker ``config.json``-style file
    (``{"auths": {"<registry>": {"auth": base64(user:password)}}}``).

    A pull through the API does not use the node's ``docker login``: the CLI reads those
    credentials, the daemon does not. Returns None (anonymous pull) without a matching entry.
    """
    if not auth_file:
        return None
    with open(auth_file) as f:
        auths = (json.load(f) or {}).get("auths") or {}
    registry = image.split("/", 1)[0] if "/" in image else "docker.io"
    for key, entry in auths.items():
        host = urllib.parse.urlsplit(key).netloc if "://" in key else key.split("/", 1)[0]
        if host != registry:
            continue
        if entry.get("auth"):
            user, _, password = base64.b64decode(entry["auth"]).decode().partition(":")
        else:
            user, password = entry.get("username", ""), entry.get("password", "")
        if not user:
            continue
        payload = {"username": user, "password": password, "serveraddress": registry}
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return None


class DockerClient:
    def __init__(self, docker_host: str | None = None, api_version: str = "1.41"):
        self.docker_host = resolve_docker_host(docker_host)
        self.api_version = api_version

    def _connection(self, timeout: float | None) -> http.client.HTTPConnection:
        url = self.docker_host
        if url.startswith("unix://"):
            return _UnixHTTPConnection(url[len("unix://") :], timeout)
        if url.startswith("/"):
            return _UnixHTTPConnection(url, timeout)
        if url.startswith("tcp://") or url.startswith("http://"):
            parts = urllib.parse.urlsplit(url.replace("tcp://", "http://", 1))
            return http.client.HTTPConnection(parts.hostname, parts.port or 2375, timeout=timeout)
        raise ValueError(f"unsupported docker_host {url!r} (unix:// or tcp://)")

    def _path(self, path: str, query: dict[str, Any] | None = None) -> str:
        full = f"/v{self.api_version}{path}"
        if query:
            full += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        return full

    def open(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = 60,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        """Send a request; return the open connection and response (caller reads and closes).
        ``body`` is JSON-encoded unless it is bytes or a file object."""
        headers = dict(headers or {})
        if body is not None and not isinstance(body, (bytes, bytearray)) and not hasattr(body, "read"):
            body = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        full = self._path(path, query)
        conn = self._connection(timeout)
        try:
            conn.request(method, full, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            raise DockerTransportError(f"{method} {full}: {e!r}") from e
        return conn, resp

    def request(self, method: str, path: str, *, ok: tuple[int, ...] = (200, 201, 204), **kwargs) -> Any:
        """One request, read in full. Returns parsed JSON (or None for an empty body); raises
        DockerAPIError on a status outside ``ok``."""
        conn, resp = self.open(method, path, **kwargs)
        try:
            data = resp.read()
        except (OSError, http.client.HTTPException) as e:
            raise DockerTransportError(f"{method} {path}: {e!r}") from e
        finally:
            conn.close()
        if resp.status not in ok:
            raise DockerAPIError(resp.status, _error_message(data), method, path)
        if not data:
            return None
        try:
            return json.loads(data)
        except ValueError:
            return data

    def version(self) -> dict[str, Any]:
        conn = self._connection(10)
        try:
            conn.request("GET", "/version")
            resp = conn.getresponse()
            return json.loads(resp.read())
        except (OSError, http.client.HTTPException) as e:
            raise DockerTransportError(f"GET /version: {e!r}") from e
        finally:
            conn.close()

    def pull(self, image: str, *, auth: str | None = None, timeout: float = 1800) -> None:
        """``POST /images/create``: the daemon streams JSON progress lines; an ``error`` line
        means the pull failed even though the status is 200."""
        name, tag = split_image(image)
        headers = {"X-Registry-Auth": auth} if auth else None
        deadline = time.monotonic() + timeout
        conn, resp = self.open(
            "POST", "/images/create", query={"fromImage": name, "tag": tag}, headers=headers, timeout=min(timeout, 300)
        )
        try:
            if resp.status != 200:
                raise DockerAPIError(resp.status, _error_message(resp.read()), "POST", f"/images/create {image}")
            pending = b""
            while True:
                if time.monotonic() > deadline:
                    raise DockerTransportError(f"pull {image} exceeded {timeout:.0f}s")
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    _raise_on_progress_error(line, image)
            _raise_on_progress_error(pending, image)
        except (OSError, http.client.HTTPException) as e:
            raise DockerTransportError(f"pull {image}: {e!r}") from e
        finally:
            conn.close()


def split_image(image: str) -> tuple[str, str]:
    """``registry:5000/a/b:tag`` -> (``registry:5000/a/b``, ``tag``); digests stay in the name."""
    if "@" in image:
        return image, ""
    last = image.rsplit("/", 1)[-1]
    if ":" in last:
        name, tag = image.rsplit(":", 1)
        return name, tag
    return image, "latest"


def _raise_on_progress_error(line: bytes, image: str) -> None:
    line = line.strip()
    if not line:
        return
    try:
        event = json.loads(line)
    except ValueError:
        return
    if isinstance(event, dict) and event.get("error"):
        raise DockerAPIError(500, str(event["error"]), "POST", f"/images/create {image}")


def _error_message(data: bytes) -> str:
    try:
        return str(json.loads(data).get("message", data[:500]))
    except (ValueError, AttributeError):
        return data[:500].decode("utf-8", "replace")

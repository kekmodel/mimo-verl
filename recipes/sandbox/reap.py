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
"""Remove leftover task sandboxes on this node's Docker daemon.

Only containers labelled ``mimo.sandbox=1`` are touched (the daemon may be shared with other
users' containers). Sandboxes exit on their own after ``max_lifetime``; this is for clearing
them sooner, e.g. after a crashed run::

    python -m recipes.sandbox.reap --exp four-whitebox --older-than 600 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .docker_api import DockerAPIError, DockerClient


def find(client: DockerClient, exp: str | None, owner: str | None, older_than: float) -> list[dict]:
    labels = ["mimo.sandbox=1"]
    if exp:
        labels.append(f"mimo.exp={exp}")
    if owner:
        labels.append(f"mimo.owner={owner}")
    containers = client.request(
        "GET", "/containers/json", query={"all": 1, "filters": json.dumps({"label": labels})}, timeout=60
    )
    cutoff = time.time() - older_than
    return [c for c in containers or [] if c.get("Created", 0) <= cutoff]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", help="only this experiment (label mimo.exp)")
    ap.add_argument("--owner", help="only sandboxes created from this host (label mimo.owner)")
    ap.add_argument("--older-than", type=float, default=0, help="seconds since creation (default 0)")
    ap.add_argument("--docker-host", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    client = DockerClient(args.docker_host)
    victims = find(client, args.exp, args.owner, args.older_than)
    failed = 0
    for c in victims:
        name = (c.get("Names") or ["?"])[0].lstrip("/")
        labels = c.get("Labels") or {}
        action = "would remove" if args.dry_run else "removing"
        print(f"{action} {name} exp={labels.get('mimo.exp')} state={c.get('State')}")
        if args.dry_run:
            continue
        try:
            client.request(
                "DELETE", f"/containers/{c['Id']}", query={"force": 1, "v": 1}, ok=(204, 404, 409), timeout=60
            )
        except DockerAPIError as e:
            failed += 1
            print(f"  failed: {e}", file=sys.stderr)
    print(f"{len(victims)} sandbox(es){' found' if args.dry_run else ' removed'}, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

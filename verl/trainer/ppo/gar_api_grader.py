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
"""GAR grader backed by an LLM API: OpenAI Chat Completions, OpenAI Responses or Anthropic Messages.

Plug it in with::

    algorithm.gar.enable=true
    algorithm.gar.grader.path=pkg://verl.trainer.ppo.gar_api_grader
    algorithm.gar.grader.name=APIGrader
    algorithm.gar.grader.kwargs.url=https://api.anthropic.com
    algorithm.gar.grader.kwargs.api=anthropic          # chat | responses | anthropic
    algorithm.gar.grader.kwargs.model=claude-opus-5

The API key is read from ``api_key_file`` or else from the trainer process's environment
variable named by ``api_key_env`` (default ``GAR_GRADER_API_KEY``; empty is allowed for a local
server), never from the config. Chat Completions servers that only accept ``max_tokens`` take
``chat_max_tokens_field=max_tokens``.

One request per mixed group. The model sees the task, every candidate's outcome, test output
and patch (failed ones as context), and returns per passing candidate the five criterion
scores of GAGAR Table 2 (1-5), the tier flags of A.1 and a confirmed-hack verdict. Tiers and
the within-tier ranking are computed here from those scores (A.1: tier rules, then weighted
score, equal scores tie), so the ranking is reproducible from the model's output.

Unlike the paper's grader this one does not enter the repository or run tests: it judges from
the patches and the test output the rollouts shipped. The Code runner ships the task text only
with ``runner_kwargs.include_task_in_reward_info=true``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from verl.trainer.ppo.gar import Candidate, Grade, Group, GroupResult

logger = logging.getLogger(__name__)

API_KINDS = ("chat", "responses", "anthropic")
_SUFFIX = {"chat": "/chat/completions", "responses": "/responses", "anthropic": "/messages"}
CRITERIA = ("approach", "precision", "minimality", "side_effects", "consistency")
WEIGHTS = {"approach": 0.30, "precision": 0.25, "minimality": 0.20, "side_effects": 0.15, "consistency": 0.10}

SYSTEM_PROMPT = """You grade code-agent solutions to one software task. Several candidates attempted \
the same task; the ones marked PASSED passed the task's tests. Compare them against each other: \
differences between implementations of the same task reveal unnecessary complexity, weakened \
validation and out-of-scope changes. Failed candidates are context only.

For every PASSED candidate, score each criterion from 1 (worst) to 5 (best):
- approach: root-cause resolution and an appropriate strategy (low: symptom-level workaround, flawed strategy)
- precision: correct change locations, complete behavior coverage (low: scattered special cases, redundant fallbacks)
- minimality: only the necessary, task-scoped edits (low: unrelated refactoring, redundant helpers or branches)
- side_effects: surrounding behavior and public interfaces preserved (low: unrelated behavior changes, new risks)
- consistency: reuses existing abstractions and conventions (low: ad hoc mechanisms, inconsistent style)

Also set, with evidence (a patch location, a trajectory event or a test result) for every true flag:
- unrequested_rewrite: the patch rewrites code the task did not ask to change
- test_specific_workaround: the patch special-cases the tests instead of fixing the behavior
- severe_process_issue: e.g. the agent disabled or edited tests, or ignored failing checks
- unresolved_regression: the patch breaks existing behavior that the tests do not cover
- hack: the solution relies on a leaked or external answer (copied fix, fetched upstream patch). \
Only mark hack when the evidence confirms it.

Reply with one JSON object and nothing else:
{"candidates": {"<id>": {"approach": int, "precision": int, "minimality": int, "side_effects": int, \
"consistency": int, "unrequested_rewrite": bool, "test_specific_workaround": bool, \
"severe_process_issue": bool, "unresolved_regression": bool, "hack": bool, "evidence": str}}}
Include exactly the PASSED candidate ids."""


def _clip(text: Any, limit: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars omitted] ...\n{text[-half:]}"


def _info(c: Candidate) -> dict:
    info = c.extra_fields.get("reward_extra_info")
    info = getattr(info, "data", info)
    merged = dict(c.extra_fields)
    if isinstance(info, dict):
        merged.update(info)
    return merged


def endpoint(url: str, api: str) -> str:
    """The request URL: ``url`` as given when it already names the endpoint, else base + path."""
    url = url.rstrip("/")
    if url.endswith(_SUFFIX[api]):
        return url
    if api == "anthropic" and not url.endswith("/v1"):
        url += "/v1"
    return url + _SUFFIX[api]


def tier_and_rank(scores: dict[str, dict]) -> dict[str, Grade]:
    """GAGAR A.1: tiers from criterion scores and flags, then weighted score order, ties kept."""
    tiers, weight = {}, {}
    for cid, s in scores.items():
        if (
            s["unrequested_rewrite"]
            or s["test_specific_workaround"]
            or s["approach"] <= 1
            or (s["minimality"] <= 2 and s["side_effects"] <= 2)
        ):
            tiers[cid] = "T3"
        elif all(s[k] >= 4 for k in CRITERIA) and not s["severe_process_issue"] and not s["unresolved_regression"]:
            tiers[cid] = "T1"
        else:
            tiers[cid] = "T2"
        weight[cid] = round(sum(WEIGHTS[k] * s[k] for k in CRITERIA), 6)
    grades = {}
    for tier in ("T1", "T2", "T3"):
        levels = sorted({weight[c] for c in tiers if tiers[c] == tier}, reverse=True)
        for c in tiers:
            if tiers[c] == tier:
                grades[c] = Grade(tier, levels.index(weight[c]))
    return grades


def parse_reply(text: str, passing_ids: list[str]) -> dict[str, dict]:
    """The JSON object in ``text``, validated against the passing ids. Raises ValueError."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(match.group(0))
    cands = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(cands, dict) or set(cands) != set(passing_ids):
        raise ValueError(f"reply grades {sorted(cands or {})}, expected {sorted(passing_ids)}")
    out = {}
    for cid, s in cands.items():
        row = {}
        for k in CRITERIA:
            v = s.get(k)
            if isinstance(v, bool) or not isinstance(v, int | float) or not 1 <= v <= 5:
                raise ValueError(f"{cid}.{k} must be a score in 1..5, got {v!r}")
            row[k] = int(round(v))
        for k in ("unrequested_rewrite", "test_specific_workaround", "severe_process_issue", "unresolved_regression", "hack"):
            row[k] = bool(s.get(k, False))
        row["evidence"] = str(s.get("evidence") or "")
        out[cid] = row
    return out


class APIGrader:
    """``grade(groups) -> {group_id: GroupResult | None}`` over an LLM API."""

    def __init__(
        self,
        url: str,
        model: str,
        api: str = "chat",
        api_key_env: str = "GAR_GRADER_API_KEY",
        api_key_file: Optional[str] = None,
        max_output_tokens: int = 4096,
        chat_max_tokens_field: str = "max_completion_tokens",
        temperature: Optional[float] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        max_workers: int = 16,
        max_patch_chars: int = 20000,
        max_text_chars: int = 4000,
        shuffle_seed: Optional[int] = 0,
        extra_body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        if api not in API_KINDS:
            raise ValueError(f"gar grader api must be one of {API_KINDS}, got {api!r}")
        if not url or not model:
            raise ValueError("gar grader needs url and model")
        self.url = endpoint(url, api)
        self.model = model
        self.api = api
        # The key never goes through the config (it would land in the resolved config and the
        # run manifest): a file readable on the trainer node, or an env var of the trainer process.
        if api_key_file:
            with open(os.path.expanduser(api_key_file)) as f:
                self.api_key = f.read().strip()
        else:
            self.api_key = os.environ.get(api_key_env, "")
        self.chat_max_tokens_field = chat_max_tokens_field
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = temperature
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.max_workers = int(max_workers)
        self.max_patch_chars = int(max_patch_chars)
        self.max_text_chars = int(max_text_chars)
        self.shuffle_seed = shuffle_seed
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(headers or {})

    # ---- request -------------------------------------------------------------------------
    def _payload(self, system: str, user: str) -> dict:
        if self.api == "chat":
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.chat_max_tokens_field: self.max_output_tokens,
            }
        elif self.api == "responses":
            body = {"model": self.model, "instructions": system, "input": user, "max_output_tokens": self.max_output_tokens}
        else:
            body = {
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": self.max_output_tokens,
            }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self.api == "anthropic":
            h["anthropic-version"] = "2023-06-01"
            if self.api_key:
                h["x-api-key"] = self.api_key
        elif self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    @staticmethod
    def reply_text(api: str, resp: dict) -> str:
        if api == "chat":
            return resp["choices"][0]["message"]["content"] or ""
        if api == "responses":
            if resp.get("output_text"):
                return resp["output_text"]
            parts = [
                c.get("text", "")
                for item in resp.get("output", [])
                for c in (item.get("content") or [])
                if c.get("type") == "output_text"
            ]
            return "".join(parts)
        return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(f"grader HTTP {e.code}: {e.read()[:500]!r}") from e
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.max_retries:
                    raise
            time.sleep(min(60.0, 2.0**attempt + random.random()))
        raise RuntimeError("unreachable")

    # ---- prompt --------------------------------------------------------------------------
    def _prompt(self, group: Group) -> tuple[str, dict[str, str]]:
        order = list(group.candidates)
        if self.shuffle_seed is not None:
            random.Random(f"{self.shuffle_seed}:{group.group_id}").shuffle(order)
        ids = {c.session_key: f"C{i + 1}" for i, c in enumerate(order)}
        task = ""
        for c in order:
            task = str(_info(c).get("task") or "")
            if task:
                break
        parts = [f"# Task\n{_clip(task, self.max_text_chars * 4) or '(task text not shipped)'}"]
        for c in order:
            info = _info(c)
            parts.append(
                f"# Candidate {ids[c.session_key]}: {'PASSED' if c.passed else 'FAILED'}\n"
                f"## Test output\n{_clip(info.get('test_output'), self.max_text_chars)}\n"
                f"## Agent's final message\n{_clip(info.get('result'), self.max_text_chars)}\n"
                f"## Patch\n{_clip(info.get('model_patch'), self.max_patch_chars) or '(no patch shipped)'}"
            )
        return "\n\n".join(parts), ids

    def grade_one(self, group: Group) -> Optional[GroupResult]:
        user, ids = self._prompt(group)
        back = {v: k for k, v in ids.items()}
        passing = [ids[c.session_key] for c in group.candidates if c.passed]
        try:
            text = self.reply_text(self.api, self._post(self._payload(SYSTEM_PROMPT, user)))
            scores = parse_reply(text, passing)
        except Exception as e:  # noqa: BLE001 - one unusable group falls back, the rest proceed
            logger.warning("[gar] group %s ungraded: %s", group.group_id, e)
            return None
        hacks = [cid for cid, s in scores.items() if s["hack"] and s["evidence"].strip()]
        grades = tier_and_rank({cid: s for cid, s in scores.items() if cid not in hacks})
        return GroupResult({back[c]: g for c, g in grades.items()}, hacks=[back[c] for c in hacks])

    def __call__(self, groups: list[Group]) -> dict[str, Optional[GroupResult]]:
        if not groups:
            return {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(groups))) as pool:
            results = list(pool.map(self.grade_one, groups))
        return {g.group_id: r for g, r in zip(groups, results, strict=True)}

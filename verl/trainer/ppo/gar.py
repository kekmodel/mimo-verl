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
"""Groupwise Advantage Redistribution (GAR), MiMo-V2.6 report 4.3.2 / GAGAR (arXiv 2609.32577).

For each mixed-outcome group, a grader ranks the passing rollouts; lower-ranked passes are
downweighted by a quality factor ``f`` and one common factor ``lambda`` restores the removed
positive advantage, so credit moves toward better passes without changing the group's total:

    a_i = r_i - mean(r)                                   (valid rollouts of the group)
    lambda = min(sum_P a_i^+ / sum_P f_i a_i^+, lambda_max)
    B_i = lambda * f_i * a_i^+  (i in P),   B_i = a_i  (i not in P)
    A_i = B_i - mean(B)                                   (paper Eq. 7 / 8)

With binary rewards ``a_i^+ = a_i`` for every pass and, below the cap, ``mean(B) = 0``.
Confirmed hacks are reset to reward 0 before the group statistics. The grader is pluggable
(``algorithm.gar.grader.{path,name,kwargs}``); an unusable result leaves the group untouched.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

TIERS = ("T1", "T2", "T3")


@dataclass
class GARConfig:
    """``algorithm.gar``. Factor defaults are the paper's Flash configuration (A.1, A.2)."""

    enable: bool = False
    f_runner: float = 0.9  # T1 candidates below the top tied group
    f_max: float = 0.85  # T2, best tied group
    f_min: float = 0.4  # T2, worst tied group
    f_low: float = 0.2  # T3
    lambda_max: float = 1.5
    # A rollout counts as passing when its outcome score (before length shaping) is >= this.
    pass_threshold: float = 1.0
    # {"path": ..., "name": ..., "kwargs": {...}}: a callable ``grade(groups) -> results``.
    grader: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("f_runner", "f_max", "f_min", "f_low"):
            v = float(getattr(self, name))
            if not (0.0 < v <= 1.0):
                raise ValueError(f"gar.{name} must be in (0, 1], got {v}")
        if self.f_min > self.f_max:
            raise ValueError(f"gar.f_min ({self.f_min}) must be <= gar.f_max ({self.f_max})")
        if not (self.lambda_max >= 1.0 and math.isfinite(self.lambda_max)):
            raise ValueError(f"gar.lambda_max must be a finite value >= 1, got {self.lambda_max}")
        if self.enable and not (self.grader.get("path") and self.grader.get("name")):
            raise ValueError("gar.enable=true needs gar.grader.path and gar.grader.name")

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["GARConfig"]:
        """None when absent or disabled."""
        if raw is None:
            return None
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(raw, DictConfig):
                raw = OmegaConf.to_container(raw, resolve=True)
        except ImportError:  # pragma: no cover
            pass
        cfg = cls(**dict(raw))
        return cfg if cfg.enable else None


@dataclass
class Candidate:
    """One valid rollout of a group, as the grader sees it."""

    session_key: str
    passed: bool
    score: float
    extra_fields: dict[str, Any]


@dataclass
class Group:
    group_id: str
    candidates: list[Candidate]


@dataclass
class Grade:
    tier: str
    rank: int  # zero-based rank of the candidate's tied group within its tier


@dataclass
class GroupResult:
    """What a grader returns for one group. ``hacks`` are session keys of confirmed hacks."""

    grades: dict[str, Grade]
    hacks: list[str] = field(default_factory=list)


Grader = Callable[[list[Group]], dict[str, Optional[GroupResult]]]


def load_grader(cfg: GARConfig) -> Grader:
    from verl.utils.import_utils import load_extern_object

    obj = load_extern_object(cfg.grader["path"], cfg.grader["name"])
    kwargs = dict(cfg.grader.get("kwargs") or {})
    return obj(**kwargs) if isinstance(obj, type) else (lambda groups: obj(groups, **kwargs))


def validate_result(group: Group, result: Any) -> Optional[str]:
    """Why ``result`` cannot be used for ``group``, or None. Type-strict: a custom grader may
    return anything, and anything but a well-formed result falls back to GRPO."""
    if result is None:
        return "no result"
    if not isinstance(result, GroupResult):
        return f"result is {type(result).__name__}, not GroupResult"
    if not isinstance(result.grades, dict) or not isinstance(result.hacks, list | tuple):
        return "grades must be a dict and hacks a list"
    keys = {c.session_key for c in group.candidates}
    passing = {c.session_key for c in group.candidates if c.passed}
    if not all(isinstance(h, str) for h in result.hacks):
        return "hack ids must be strings"
    hacks = set(result.hacks)
    if not hacks <= passing:
        return "hack outside the passing candidates"
    for key, grade in result.grades.items():
        if key not in keys:
            return f"grade for unknown candidate {key}"
        if not isinstance(grade, Grade) or grade.tier not in TIERS:
            return f"bad grade {grade!r}"
        if isinstance(grade.rank, bool) or not isinstance(grade.rank, int) or grade.rank < 0:
            return f"bad rank {grade.rank!r}"
    missing = passing - hacks - set(result.grades)
    if missing:
        return f"passing candidates missing from the ranking: {sorted(missing)}"
    return None


def factors(grades: dict[str, Grade], cfg: GARConfig) -> dict[str, float]:
    """Tier / tied-group rank -> quality factor (paper Eq. 6)."""
    t2_ranks = sorted({int(g.rank) for g in grades.values() if g.tier == "T2"})
    k2 = len(t2_ranks)
    t2_pos = {r: i for i, r in enumerate(t2_ranks)}  # ranks may skip numbers; use tied-group order
    t1_top = min((int(g.rank) for g in grades.values() if g.tier == "T1"), default=None)
    out = {}
    for key, g in grades.items():
        if g.tier == "T1":
            out[key] = 1.0 if int(g.rank) == t1_top else cfg.f_runner
        elif g.tier == "T2":
            out[key] = cfg.f_max if k2 == 1 else cfg.f_max - (cfg.f_max - cfg.f_min) * t2_pos[int(g.rank)] / (k2 - 1)
        else:
            out[key] = cfg.f_low
    return out


def redistribute(
    scores: Sequence[float],
    passed: Sequence[bool],
    f: Sequence[Optional[float]],
    lambda_max: float,
) -> tuple[list[float], dict[str, float]]:
    """Group advantages after redistribution (paper Eq. 7 / 8).

    ``scores`` are the valid rollouts' rewards after any shaping, ``passed`` their outcome,
    ``f`` their factors (None for failures). Returns advantages with zero group mean.
    """
    n = len(scores)
    mean = sum(scores) / n
    a = [s - mean for s in scores]
    pos = [max(x, 0.0) if p else 0.0 for x, p in zip(a, passed, strict=True)]
    num = sum(pos)
    den = sum((fi or 0.0) * x for fi, x in zip(f, pos, strict=True))
    info = {"capped": 0.0, "lambda": 1.0}
    if den <= 0.0:
        return a, info
    lam = num / den
    if lam > lambda_max:
        info["capped"] = 1.0
        lam = lambda_max
    info["lambda"] = lam
    b = [lam * fi * x if p else ai for ai, x, fi, p in zip(a, pos, f, passed, strict=True)]
    mb = sum(b) / n
    return [x - mb for x in b], info


def _meta(value: Any) -> dict:
    value = getattr(value, "data", value)
    return value if isinstance(value, dict) else {}


class GARStep:
    """The two trainer hooks of one step: grade before GRPO, redistribute after it.

    Rows are GRPO rows (``{uid}_{session}_{index}`` batch keys); a session is graded on its
    final row, the one GRPO uses, and its new advantage is broadcast to all its rows.
    """

    def __init__(self, cfg: GARConfig, grader: Grader):
        self.cfg = cfg
        self.grader = grader
        self._groups: dict[str, list[str]] = {}  # group id -> session keys, graded groups only
        self._passed: dict[str, bool] = {}
        self._f: dict[str, float] = {}

    def grade(self, token_level_rewards, group_ids, batch_keys, invalid, extra_fields):
        """Call the grader on mixed groups. Returns ``(rewards, metrics)``: a copy of
        ``token_level_rewards`` with confirmed hacks reset to 0, and the metrics."""
        from verl.trainer.ppo.advantage_fixes import final_rows_by_session

        final = final_rows_by_session(batch_keys)
        scores = token_level_rewards.sum(dim=-1)
        members: dict[str, list[str]] = {}
        for skey, row in final.items():
            if not bool(invalid[row]):
                members.setdefault(str(group_ids[row]), []).append(skey)
        groups = []
        for gid, skeys in members.items():
            passed = [float(scores[final[s]]) >= self.cfg.pass_threshold for s in skeys]
            if len(skeys) >= 2 and any(passed) and not all(passed):
                groups.append(
                    Group(
                        gid,
                        [
                            Candidate(s, p, float(scores[final[s]]), _meta(extra_fields[final[s]]) if extra_fields else {})
                            for s, p in zip(skeys, passed, strict=True)
                        ],
                    )
                )
        metrics = {"gar/groups_eligible": float(len(groups)), "gar/groups_graded": 0.0, "gar/groups_fallback": 0.0}
        self._groups, self._passed, self._f = {}, {}, {}
        if not groups:
            return token_level_rewards, metrics
        try:
            results = self.grader(groups)
            if results is None:
                results = {}
            if not isinstance(results, dict):
                raise TypeError(f"grader returned {type(results).__name__}, not a dict")
        except Exception as e:  # noqa: BLE001 - an unusable grade falls back to GRPO, like an unusable result
            logger.warning("[gar] grader failed on %d groups, falling back to GRPO: %s", len(groups), e)
            metrics["gar/grader_error"] = 1.0
            metrics["gar/groups_fallback"] = float(len(groups))
            return token_level_rewards, metrics
        rewards = token_level_rewards.clone()  # hacks are zeroed on a copy, never on rm_scores
        tiers = {t: 0 for t in TIERS}
        hacks = grades_on_failures = 0
        for group in groups:
            result = results.get(group.group_id)
            reason = validate_result(group, result)
            if reason is None:
                try:
                    hacked = set(result.hacks)
                    passing = {c.session_key for c in group.candidates if c.passed} - hacked
                    # Only passing candidates are ranked (paper 3.2); a grade on a failure must not
                    # move the T1 top or the T2 tied-group count.
                    grades_on_failures += sum(1 for k in result.grades if k not in passing and k not in hacked)
                    f = factors({k: g for k, g in result.grades.items() if k in passing}, self.cfg)
                except Exception as e:  # noqa: BLE001
                    reason = f"{type(e).__name__}: {e}"
            if reason is not None:
                logger.warning("[gar] group %s falls back to GRPO: %s", group.group_id, reason)
                metrics["gar/groups_fallback"] += 1
                continue
            metrics["gar/groups_graded"] += 1
            self._groups[group.group_id] = [c.session_key for c in group.candidates]
            for c in group.candidates:
                self._passed[c.session_key] = c.session_key in passing
                if c.session_key in passing:
                    self._f[c.session_key] = f[c.session_key]
                    tiers[result.grades[c.session_key].tier] += 1
                if c.session_key in hacked:
                    rewards[final[c.session_key]] = 0.0
                    hacks += 1
        graded = sum(tiers.values())
        metrics["gar/confirmed_hacks"] = float(hacks)
        metrics["gar/grades_on_failures"] = float(grades_on_failures)
        for t, count in tiers.items():
            metrics[f"gar/tier_share_{t}"] = count / graded if graded else 0.0
        return rewards, metrics

    def redistribute(self, advantages, response_mask, token_level_rewards, batch_keys) -> tuple[Any, dict[str, float]]:
        """Replace graded groups' GRPO advantages with the redistributed ones."""
        from verl.trainer.ppo.advantage_fixes import final_rows_by_session, session_keys_from_batch_keys

        if not self._groups:
            return advantages, {}
        final = final_rows_by_session(batch_keys)
        scores = token_level_rewards.sum(dim=-1)
        rows_of: dict[str, list[int]] = {}
        for row, skey in enumerate(session_keys_from_batch_keys(batch_keys)):
            rows_of.setdefault(skey, []).append(row)
        lambdas, capped = [], []
        for skeys in self._groups.values():
            new, info = redistribute(
                [float(scores[final[s]]) for s in skeys],
                [self._passed[s] for s in skeys],
                [self._f.get(s) for s in skeys],
                self.cfg.lambda_max,
            )
            lambdas.append(info["lambda"])
            capped.append(info["capped"])
            for s, value in zip(skeys, new, strict=True):
                for row in rows_of[s]:
                    advantages[row] = value * response_mask[row].to(advantages.dtype)
        return advantages, {
            "gar/lambda_mean": sum(lambdas) / len(lambdas),
            "gar/lambda_capped_rate": sum(capped) / len(capped),
        }

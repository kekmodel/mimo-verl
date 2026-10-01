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
"""Sample Mixer (MiMo-V2.6 report 6.3): fill every training batch with a target mix of data
sources although their rollout durations and acceptance rates differ by orders of magnitude.

Per source ``i``: ``B_i`` accepted groups per training batch (the quota), ``r_i`` the group
acceptance rate (dynamic sampling keeps only groups with reward variance), ``t_i`` the active
rollout duration (generation + environment, *without* the colocated trainer's pauses).

* Quotas. ``target_basis=accepted`` (the report): ``B_i = π_i·B`` accepted groups, so a source is
  weighted ``1/r_i`` relative to its share of generated prompts (zero-variance groups carry no
  gradient). ``target_basis=generated``: ``B_i ∝ π_i·r̂_i``, which makes the expected gradient
  ``Σ π_i ∇J_i`` over generated prompts.
* Adaptive Rollout Concurrency (Eq. 6): demand ``m_i = B_i / r_i``; scheduling budget
  ``(1 + p_i)·m_i`` groups with ``p_i = clip(c·t_i − 1, p_min, p_max)`` and ``c`` set (bisection,
  monotone) so the demand-weighted mean of ``p_i`` is ``p_mean``.
* Adaptive Rollout Scheduling (Eq. 7): each prompt fetch picks a source by smooth weighted
  round-robin over ``w_i = α·B_i/r_i + (1−α)·(B_i − A_i)⁺/r_i`` among sources whose in-flight
  plus accepted-but-unconsumed groups are below budget (``A_i``: accepted groups waiting).
  If every source is at budget the pick ignores budgets (counted): the trainer feeds one prompt
  per consumed or rejected group, so total inflow equals outflow and nothing accumulates.
* Batch assembly (``MixerReplayBuffer``): a batch takes exactly ``B_i`` accepted groups per
  source, oldest first; surplus waits for the next batch and is never dropped (dropping it
  would censor the slow, long rollouts).

Predictive rollout dispatch (KV-aware placement, an inference-engine concern) and sample
replay are not implemented.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MixerConfig:
    """``trainer.v1.sampler.mixer``."""

    enable: bool = False
    # name -> {data_sources: [...], weight: float, prior_accept: float, prior_duration: float}
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_basis: str = "accepted"  # accepted | generated
    alpha: float = 0.5
    p_mean: float = 1.0
    p_min: float = 0.0
    p_max: float = 4.0
    ema: float = 0.1
    # Until a source has finished groups, allocate by priors ∝ t_i·m_i (report: steady-state startup).
    steady_state_startup: bool = True

    def __post_init__(self):
        if self.target_basis not in ("accepted", "generated"):
            raise ValueError(f"mixer.target_basis must be accepted or generated, got {self.target_basis!r}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("mixer.alpha must be in [0, 1]")
        if not self.p_min <= self.p_mean <= self.p_max:
            raise ValueError("mixer needs p_min <= p_mean <= p_max")
        if not 0.0 < self.ema <= 1.0:
            raise ValueError("mixer.ema must be in (0, 1]")
        if self.enable:
            if not self.sources:
                raise ValueError("mixer.enable needs mixer.sources")
            seen: dict[str, str] = {}
            for name, spec in self.sources.items():
                if float(spec.get("weight", 0.0)) <= 0.0:
                    raise ValueError(f"mixer.sources.{name}.weight must be positive")
                for ds in spec.get("data_sources") or []:
                    if ds in seen:
                        raise ValueError(f"data_source {ds!r} is in both {seen[ds]} and {name}")
                    seen[ds] = name
                if not spec.get("data_sources"):
                    raise ValueError(f"mixer.sources.{name}.data_sources is empty")

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["MixerConfig"]:
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


def apportion(weights: dict[str, float], total: int) -> dict[str, int]:
    """Integer split of ``total`` proportional to ``weights`` (largest remainder, ties by name)."""
    s = sum(weights.values())
    if total <= 0 or s <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * w / s for k, w in weights.items()}
    out = {k: math.floor(v) for k, v in exact.items()}
    rest = total - sum(out.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


def oversampling(demand: dict[str, float], duration: dict[str, float], p_mean: float, p_min: float, p_max: float) -> dict[str, float]:
    """Eq. 6: ``p_i = clip(c·t_i − 1, p_min, p_max)`` with ``Σ m_i p_i / Σ m_i = p_mean``."""
    total = sum(demand.values())
    if total <= 0:
        return {k: p_mean for k in demand}

    def mean_p(c: float) -> float:
        return sum(m * min(max(c * duration[k] - 1.0, p_min), p_max) for k, m in demand.items()) / total

    lo, hi = 0.0, 1.0
    t_min = min((t for t in duration.values() if t > 0), default=1.0)
    hi = max(hi, (p_max + 1.0) / t_min)
    if mean_p(lo) >= p_mean:
        c = lo
    else:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if mean_p(mid) < p_mean:
                lo = mid
            else:
                hi = mid
        c = hi
    return {k: min(max(c * duration[k] - 1.0, p_min), p_max) for k in demand}


@dataclass
class _Group:
    source: str
    submitted: float
    state: str = "inflight"  # inflight | accepted


class SampleMixer:
    """Driver-side state of the mixer. ``clock`` is injectable for tests and simulation."""

    def __init__(self, cfg: MixerConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.names = list(cfg.sources)
        self.pi = {n: float(cfg.sources[n]["weight"]) for n in self.names}
        z = sum(self.pi.values())
        self.pi = {n: w / z for n, w in self.pi.items()}
        self.by_data_source = {ds: n for n, spec in cfg.sources.items() for ds in spec["data_sources"]}
        self.r = {n: float(cfg.sources[n].get("prior_accept", 0.5)) for n in self.names}
        self.t = {n: float(cfg.sources[n].get("prior_duration", 1.0)) for n in self.names}
        self.seen_terminal = {n: 0 for n in self.names}
        self.groups: dict[str, _Group] = {}
        self._swrr = {n: 0.0 for n in self.names}
        self._pauses: list[list[float]] = []  # [start, end or None]
        self._counters: dict[str, float] = defaultdict(float)
        self.batch_size = 0

    # ---- sources ---------------------------------------------------------------------------
    def source_of_data_source(self, data_source: str) -> str:
        if data_source not in self.by_data_source:
            raise ValueError(f"data_source {data_source!r} belongs to no mixer source; known: {sorted(self.by_data_source)}")
        return self.by_data_source[data_source]

    # ---- quotas, demand, budgets -----------------------------------------------------------
    def quotas(self, batch_size: int) -> dict[str, int]:
        self.batch_size = batch_size
        if self.cfg.target_basis == "accepted":
            return apportion(self.pi, batch_size)
        return apportion({n: self.pi[n] * max(self.r[n], 1e-3) for n in self.names}, batch_size)

    def demand(self, batch_size: int) -> dict[str, float]:
        q = self.quotas(batch_size)
        return {n: q[n] / max(self.r[n], 1e-3) for n in self.names}

    def budgets(self, batch_size: int) -> dict[str, float]:
        m = self.demand(batch_size)
        p = oversampling(m, self.t, self.cfg.p_mean, self.cfg.p_min, self.cfg.p_max)
        return {n: (1.0 + p[n]) * m[n] for n in self.names}

    def counts(self) -> tuple[dict[str, int], dict[str, int]]:
        inflight = {n: 0 for n in self.names}
        accepted = {n: 0 for n in self.names}
        for g in self.groups.values():
            (inflight if g.state == "inflight" else accepted)[g.source] += 1
        return inflight, accepted

    # ---- scheduling (Eq. 7) ----------------------------------------------------------------
    def weights(self, batch_size: int) -> dict[str, float]:
        q = self.quotas(batch_size)
        _, accepted = self.counts()
        if self.cfg.steady_state_startup and not any(self.seen_terminal.values()):
            return {n: self.t[n] * q[n] / max(self.r[n], 1e-3) for n in self.names}
        a = self.cfg.alpha
        return {
            n: a * q[n] / max(self.r[n], 1e-3) + (1 - a) * max(q[n] - accepted[n], 0) / max(self.r[n], 1e-3)
            for n in self.names
        }

    def choose_source(self, batch_size: int) -> str:
        w = self.weights(batch_size)
        budgets = self.budgets(batch_size)
        inflight, accepted = self.counts()
        eligible = [n for n in self.names if w[n] > 0 and inflight[n] + accepted[n] < budgets[n]]
        if not eligible:
            self._counters["over_budget_picks"] += 1
            eligible = [n for n in self.names if w[n] > 0] or list(self.names)
            if all(w[n] <= 0 for n in eligible):
                w = {n: self.pi[n] for n in self.names}
        total = sum(w[n] for n in eligible)
        for n in eligible:
            self._swrr[n] += w[n]
        pick = max(eligible, key=lambda n: (self._swrr[n], -self.names.index(n)))
        self._swrr[pick] -= total
        return pick

    # ---- group lifecycle -------------------------------------------------------------------
    def on_submit(self, uid: str, source: str, now: Optional[float] = None) -> None:
        self.groups[uid] = _Group(source, self.clock() if now is None else now)

    def _active_duration(self, g: _Group, now: float) -> float:
        paused = 0.0
        for start, end in self._pauses:
            end = now if end is None else end
            paused += max(0.0, min(end, now) - max(start, g.submitted))
        return max(0.0, now - g.submitted - paused)

    def _finish(self, uid: str, accepted: bool) -> None:
        g = self.groups.get(uid)
        if g is None or g.state != "inflight":
            return
        now = self.clock()
        e = self.cfg.ema
        n = g.source
        self.r[n] = (1 - e) * self.r[n] + e * (1.0 if accepted else 0.0)
        self.t[n] = (1 - e) * self.t[n] + e * self._active_duration(g, now)
        self.seen_terminal[n] += 1
        if accepted:
            g.state = "accepted"
        else:
            del self.groups[uid]

    def on_accepted(self, uid: str) -> None:
        self._finish(uid, True)

    def on_rejected(self, uid: str) -> None:
        self._finish(uid, False)

    def on_consumed(self, uids) -> None:
        for uid in uids:
            self.groups.pop(uid, None)

    def pause(self) -> None:
        if not self._pauses or self._pauses[-1][1] is not None:
            self._pauses.append([self.clock(), None])

    def resume(self) -> None:
        if self._pauses and self._pauses[-1][1] is None:
            self._pauses[-1][1] = self.clock()
        # Pauses older than every live group no longer matter.
        oldest = min((g.submitted for g in self.groups.values()), default=self.clock())
        self._pauses = [p for p in self._pauses if p[1] is None or p[1] > oldest]

    # ---- reporting / checkpoint ------------------------------------------------------------
    def metrics(self, batch_size: int) -> dict[str, float]:
        q = self.quotas(batch_size)
        budgets = self.budgets(batch_size)
        inflight, accepted = self.counts()
        eff = {n: q[n] / max(self.r[n], 1e-3) for n in self.names}
        z = sum(eff.values()) or 1.0
        out = {}
        for n in self.names:
            p = f"mixer/{n}"
            out.update(
                {
                    f"{p}/quota": float(q[n]),
                    f"{p}/accept_rate": self.r[n],
                    f"{p}/active_duration_s": self.t[n],
                    f"{p}/budget": budgets[n],
                    f"{p}/inflight": float(inflight[n]),
                    f"{p}/accepted_waiting": float(accepted[n]),
                    # Share of the gradient over generated prompts this quota implies: the
                    # accepted basis weights a source by B_i / r_i.
                    f"{p}/generated_share": eff[n] / z,
                }
            )
        out["mixer/over_budget_picks"] = self._counters["over_budget_picks"]
        return out

    def state_dict(self) -> dict:
        return {"r": dict(self.r), "t": dict(self.t), "seen_terminal": dict(self.seen_terminal), "swrr": dict(self._swrr)}

    def load_state_dict(self, state: dict) -> None:
        for key, target in (("r", self.r), ("t", self.t), ("seen_terminal", self.seen_terminal), ("swrr", self._swrr)):
            for n, v in (state.get(key) or {}).items():
                if n in target:
                    target[n] = v

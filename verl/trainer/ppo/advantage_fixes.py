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
"""Row-level corrections for GRPO + prompt-mean training on agentic rollouts.

The target objective under ``loss_agg_mode=prompt-mean`` is

    L = (1/|Q|) * sum_q (1/T_q) * sum_{i in V_q} sum_t l_{i,t}

where Q are prompts with at least one valid row, V_q the valid rows of prompt q and T_q
their action-token count. A row invalidated by infrastructure (pod died, setup failed,
verifier transport error) is not a sample of the policy and must appear in none of Q, V_q,
T_q or the GRPO baseline. The helpers here implement that exactly:

* :func:`invalid_rows` / :func:`expand_invalid_to_sessions` — one boolean per row from
  ``is_infra`` metadata or the ``invalid_reward_value`` sentinel; a session graded on an
  invalid final row is invalid as a whole.
* :func:`isolate_invalid_rows` — before GRPO, move invalid rows into a group of their own, so
  each group's baseline (mean, and std when normalizing) is over its valid sessions.
* :func:`group_size_correction` — GRPO's baseline includes the row itself, so
  ``E[(r_i - mean) * grad log pi_i] = (1 - 1/n) * grad J_q``. With a fixed group size this is a
  common constant; once rows drop out, ``n`` differs by group and the factor becomes a
  per-prompt bias. Rescaling each group by ``(1 - 1/n_ref) / (1 - 1/n_valid)`` restores a
  common factor (= RLOO x (1 - 1/n_ref)) without changing full groups.
* :func:`mask_invalid_rows` — zero the advantage and the loss mask of invalid rows, so
  :func:`verl.trainer.ppo.core_algos.compute_prompt_loss_weights` counts neither their
  tokens nor their prompt.
* :func:`session_length_signals` / :func:`apply_group_length_penalty` — the group-relative
  length penalty (report Eq. 4, reference profile values) on any recipe, from per-row
  tensors when the agent loop ships no ``length_signals``.
* :func:`tool_error_hits_from_spans` — the segment-level tool-error mask built from
  ``llm_turn_spans`` / ``tool_call_error_flags`` metadata, for agent loops that do not ship
  a ``tool_call_error_mask`` tensor.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch

from verl.utils.length_penalty import LengthPenaltyConfig, compute_group_length_penalty, finalize_length_penalty_metrics


_LEGACY_LENGTH_PENALTY_KEYS = {"enable": "enabled", "deadzone": "excess_threshold", "saturate": "excess_saturate"}


def length_penalty_config(raw: Mapping[str, Any] | None) -> LengthPenaltyConfig | None:
    """Build a :class:`LengthPenaltyConfig`, accepting the recipes' legacy key names.

    ``recipes/general/config/general.yaml`` spells the fields ``enable`` / ``deadzone`` /
    ``saturate``; before this module nothing in the trainer read ``algorithm.length_penalty``
    at all, so that block was silently inert.
    """
    if raw is None:
        return None
    values = {}
    for key, value in dict(raw).items():
        target = _LEGACY_LENGTH_PENALTY_KEYS.get(key, key)
        if target in values:
            raise ValueError(f"length_penalty sets {target!r} twice (legacy and current key names)")
        values[target] = value
    if "metrics" in values and values["metrics"] is not None:
        values["metrics"] = tuple(values["metrics"])
    cfg = LengthPenaltyConfig(**values)
    return cfg if cfg.enabled else None


def _meta(value: Any) -> dict:
    value = getattr(value, "data", value)
    return value if isinstance(value, dict) else {}


def exec_budget_hit(extra_field: Any) -> float:
    """1.0 if the rollout was frozen at its tool-execution budget (general: extra_fields;
    code: the runner's reward_info, carried as reward_extra_info)."""
    info = _meta(extra_field)
    value = info.get("exec_budget_hit")
    if value is None:
        value = _meta(info.get("reward_extra_info")).get("exec_budget_hit", 0.0)
    return 1.0 if float(value or 0.0) > 0.5 else 0.0


def exec_budget_hit_rate(
    extra_fields: Sequence[Any], batch_keys: Sequence[str], invalid: torch.Tensor
) -> Optional[float]:
    """Share of valid sessions frozen at their tool-execution budget (read on the final row)."""
    rows = [row for row in final_rows_by_session(batch_keys).values() if not bool(invalid[row])]
    if not rows:
        return None
    return float(np.mean([exec_budget_hit(extra_fields[row]) for row in rows]))


def invalid_rows(
    extra_fields: Sequence[Any] | None,
    scores: torch.Tensor,
    invalid_reward_value: float | None,
) -> torch.Tensor:
    """Rows that are not samples of the policy: ``is_infra`` set, or score == sentinel."""
    invalid = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    if extra_fields is not None:
        if len(extra_fields) != scores.shape[0]:
            raise ValueError(f"extra_fields has {len(extra_fields)} rows, scores has {scores.shape[0]}")
        flags = [float(_meta(ef).get("is_infra", 0.0) or 0.0) > 0.5 for ef in extra_fields]
        invalid |= torch.tensor(flags, dtype=torch.bool, device=scores.device)
    if invalid_reward_value is not None:
        invalid |= scores == float(invalid_reward_value)
    return invalid


def _last_valid_positions(response_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(response_mask.shape[1], device=response_mask.device)
    return torch.where(response_mask.bool(), positions, -1).amax(dim=-1)


def set_row_scores(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, rows, values) -> torch.Tensor:
    """Write scalar ``values`` as the outcome reward of ``rows`` (on the last response token)."""
    out = token_level_rewards.clone()
    ends = _last_valid_positions(response_mask)
    for row, value in zip(rows, values, strict=True):
        out[row] = 0.0
        end = int(ends[row])
        out[row, end if end >= 0 else out.shape[1] - 1] = float(value)
    return out


def final_rows_by_session(batch_keys: Sequence[str]) -> dict[str, int]:
    """``{uid}_{session}`` -> row index of that session's last output.

    ``compute_advantage_for_multi_trajectories`` computes GRPO on exactly these rows (one per
    session) and broadcasts the result to the session's other rows, so anything that must
    agree with the GRPO baseline has to be computed over them too.
    """
    final: dict[str, tuple[int, int]] = {}
    for row, key in enumerate(batch_keys):
        uid, session, index = key.rsplit("_", 2)
        skey = f"{uid}_{session}"
        if skey not in final or final[skey][0] < int(index):
            final[skey] = (int(index), row)
    return {skey: row for skey, (_, row) in final.items()}


def expand_invalid_to_sessions(invalid: torch.Tensor, batch_keys: Sequence[str]) -> torch.Tensor:
    """Invalidity is a session property: GRPO grades a session on its final row, so a session
    whose final row is invalid is not a sample of the policy and all its rows are invalid. A row
    that is invalid itself stays invalid (its session keeps its other rows)."""
    final = final_rows_by_session(batch_keys)
    bad = {skey for skey, row in final.items() if bool(invalid[row])}
    out = invalid.clone()
    for row, skey in enumerate(session_keys_from_batch_keys(batch_keys)):
        if skey in bad:
            out[row] = True
    return out


def isolate_invalid_rows(uids: np.ndarray, invalid: torch.Tensor, batch_keys: Sequence[str]) -> np.ndarray:
    """uids with every invalid row moved to a GRPO group of its own session.

    The group's baseline (mean and, with std normalization, std) is then computed over the
    valid sessions only, for every estimator variant; the isolated rows are masked afterwards
    (``mask_invalid_rows``), so they carry no gradient and, having no loss tokens, count as no
    prompt in the prompt-mean normalization.
    """
    out = np.array(uids, dtype=object, copy=True)
    for row, skey in enumerate(session_keys_from_batch_keys(batch_keys)):
        if bool(invalid[row]):
            out[row] = f"__invalid__{skey}"
    return out


def session_keys_from_batch_keys(batch_keys: Sequence[str]) -> list[str]:
    """``{uid}_{session}_{output}`` -> ``{uid}_{session}`` (one GRPO sample per session)."""
    out = []
    for key in batch_keys:
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Unexpected batch key format: {key}")
        out.append(f"{parts[0]}_{parts[1]}")
    return out


def group_size_correction(
    advantages: torch.Tensor,
    group_ids: np.ndarray,
    batch_keys: Sequence[str],
    invalid: torch.Tensor,
    n_ref: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rescale each GRPO group by ``(1 - 1/n_ref) / (1 - 1/n_valid)``.

    GRPO's mean baseline includes the sample itself, so its expected gradient is
    ``(1 - 1/n) grad J`` for a group of ``n``; this makes the factor ``(1 - 1/n_ref)`` for every
    group, i.e. each group's advantage becomes ``(1 - 1/n_ref)`` times the leave-one-out (RLOO)
    advantage. Identity when ``n_valid == n_ref``; with harness grouping, harness subgroups of a
    full prompt group are smaller than ``n_ref`` and are rescaled too. ``n_valid`` counts valid
    sessions by their final row (the row GRPO grades). A group with one valid session has no
    baseline (GRPO gives it its raw reward) and is zeroed, as dynamic sampling does for a group
    without reward variance. ``n_ref < 2`` leaves the advantages unchanged.
    """
    if n_ref < 2:
        return advantages, {"training/group_size/skipped_n_ref": 1.0}
    n_valid: dict[Any, int] = defaultdict(int)
    for row in final_rows_by_session(batch_keys).values():
        if not bool(invalid[row]):
            n_valid[group_ids[row]] += 1
    base = 1.0 - 1.0 / n_ref
    factors = torch.ones(advantages.shape[0], dtype=advantages.dtype, device=advantages.device)
    for i, gid in enumerate(group_ids.tolist()):
        n = n_valid.get(gid, 0)
        if n == 1:
            factors[i] = 0.0
        elif n >= 2 and n != n_ref:
            factors[i] = base / (1.0 - 1.0 / n)
    metrics = {
        "training/group_size/corrected_groups": float(sum(1 for n in n_valid.values() if n >= 2 and n != n_ref)),
        "training/group_size/singleton_groups": float(sum(1 for n in n_valid.values() if n == 1)),
        "training/group_size/factor_max": float(factors.max().item()) if factors.numel() else 1.0,
    }
    return advantages * factors.unsqueeze(-1), metrics


def mask_invalid_rows(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    invalid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Zero advantage and loss mask of invalid rows.

    Returns ``(advantages, response_mask, edited)``. If every row is invalid the loss mask is
    left intact (advantages are already 0, so the step contributes no gradient) because
    prompt-mean weights need at least one prompt with action tokens.
    """
    if not bool(invalid.any()):
        return advantages, response_mask, False
    keep = (~invalid).to(advantages.dtype).unsqueeze(-1)
    advantages = advantages * keep
    if bool(invalid.all()):
        return advantages, response_mask, False
    response_mask = response_mask * (~invalid).to(response_mask.dtype).unsqueeze(-1)
    return advantages, response_mask, True


def action_segments(response_mask: torch.Tensor) -> torch.Tensor:
    """Per row, the number of contiguous runs of action tokens: one per model turn, since each
    turn's generated tokens are followed by tool output (mask 0)."""
    m = response_mask.to(torch.bool)
    starts = m[:, :1].to(torch.long).sum(dim=-1) + (m[:, 1:] & ~m[:, :-1]).to(torch.long).sum(dim=-1)
    return starts


def session_length_signals(
    extra_fields: Sequence[Any] | None,
    action_mask: torch.Tensor,
    prompt_lengths: Sequence[int],
    response_lengths: Sequence[int],
    session_keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-session ``{turn_count, prefill_length, decode_length}``.

    Prefers the agent loop's ``length_signals`` (model turns, prompt + tool tokens, generated
    tokens). Otherwise derives them from ``action_mask``, the response mask as the rollout
    produced it (before any penalty edits it): decode = action tokens, prefill = prompt +
    non-action response tokens, turns = runs of action tokens (model calls, the same quantity
    the agent loops report). Signals are summed over a session's output rows (sub-agents /
    compaction segments).
    """
    decode = action_mask.to(torch.bool).sum(dim=-1).tolist()
    turns = action_segments(action_mask).tolist()
    out: dict[str, dict[str, float]] = {}
    for i, skey in enumerate(session_keys):
        ls = _meta(extra_fields[i]).get("length_signals") if extra_fields is not None else None
        if isinstance(ls, dict) and all(k in ls for k in ("turn_count", "prefill_length", "decode_length")):
            row = {k: float(ls[k]) for k in ("turn_count", "prefill_length", "decode_length")}
        else:
            row = {
                "turn_count": float(turns[i]),
                "prefill_length": float(prompt_lengths[i]) + float(response_lengths[i]) - float(decode[i]),
                "decode_length": float(decode[i]),
            }
        acc = out.setdefault(skey, {"turn_count": 0.0, "prefill_length": 0.0, "decode_length": 0.0})
        for k, v in row.items():
            acc[k] += v
    return out


def apply_group_length_penalty(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uids: np.ndarray,
    session_keys: Sequence[str],
    invalid: torch.Tensor,
    signals: Mapping[str, Mapping[str, float]],
    cfg: LengthPenaltyConfig,
    batch_keys: Sequence[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Report Eq. 4 on raw outcome scores, per prompt group, valid sessions only.

    A session's score is read from its final row (the one GRPO uses) and the shaped score is
    written back to every row of the session.
    """
    if not cfg.enabled:
        return token_level_rewards, {}
    scores = token_level_rewards.sum(dim=-1)
    session_rows: dict[str, list[int]] = defaultdict(list)
    for i, skey in enumerate(session_keys):
        session_rows[skey].append(i)
    finals = final_rows_by_session(batch_keys) if batch_keys is not None else {k: v[-1] for k, v in session_rows.items()}
    groups: dict[Any, list[str]] = defaultdict(list)
    for skey, rows in session_rows.items():
        if any(bool(invalid[r]) for r in rows):
            continue
        uid = uids[rows[0]]
        if skey not in groups[uid]:
            groups[uid].append(skey)
    accumulated: dict[str, float] = {}
    rows_to_set: list[int] = []
    values: list[float] = []
    for skeys in groups.values():
        raw = [float(scores[finals[s]]) for s in skeys]
        deltas, stats = compute_group_length_penalty(raw, [signals.get(s) for s in skeys], cfg)
        for key, value in stats.items():
            accumulated[key] = accumulated.get(key, 0.0) + value
        for s, r, d in zip(skeys, raw, deltas, strict=True):
            if d != 0.0:
                for row in session_rows[s]:
                    rows_to_set.append(row)
                    values.append(r + d)
    metrics = finalize_length_penalty_metrics(accumulated, cfg.metrics)
    if not rows_to_set:
        return token_level_rewards, metrics
    return set_row_scores(token_level_rewards, response_mask, rows_to_set, values), metrics


def tool_error_hits_from_spans(
    extra_fields: Sequence[Any],
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Boolean ``[B, L]`` mask of model turns whose tool call failed.

    Reads ``llm_turn_spans`` (response-token offsets of each model turn) and
    ``tool_call_error_flags`` (one bool per turn). Fails closed per row: missing, misaligned or
    out-of-range metadata marks nothing on that row and is counted in the metrics.
    """
    mask = torch.zeros_like(response_mask, dtype=torch.bool)
    misaligned = 0
    missing = 0
    width = response_mask.shape[1]
    for row, ef in enumerate(extra_fields):
        info = _meta(ef)
        spans, flags = info.get("llm_turn_spans"), info.get("tool_call_error_flags")
        if spans is None or flags is None:
            # No metadata at all: the recipe does not record it, so the penalty is off for this
            # row. Counted so a wrong mask_source shows up instead of silently doing nothing.
            missing += int(bool(response_mask[row].any()))
            continue
        if not spans:
            continue
        if len(spans) != len(flags):
            misaligned += 1
            continue
        prev_end, bad = 0, False
        for span in spans:
            start, end = int(span[0]), int(span[1])
            if not (prev_end <= start <= end):
                bad = True
                break
            prev_end = end
        if bad:
            misaligned += 1
            continue
        for (start, end), flag in zip(spans, flags, strict=True):
            if flag:
                s, e = max(0, int(start)), min(width, int(end))
                if s < e:
                    mask[row, s:e] = True
    mask &= response_mask.to(torch.bool)
    return mask, {
        "penalty/tool_call_error_span_misaligned_rows": float(misaligned),
        "penalty/tool_call_error_span_missing_rows": float(missing),
    }


def check_policy_loss_config(config) -> None:
    """Fail fast when the rollout-correction mode and the actor loss disagree.

    ``algorithm.rollout_correction.bypass_mode`` makes the trainer use the rollout engine's
    log-probs as the old policy, and only ``actor.policy_loss.loss_mode=bypass_mode`` computes
    the loss against them (REINFORCE or PPO-clip, per ``rollout_correction.loss_type``). Either
    one without the other trains on the wrong ratio without any error.
    """
    from omegaconf import OmegaConf

    bypass = bool(OmegaConf.select(config, "algorithm.rollout_correction.bypass_mode", default=False))
    loss_mode = str(OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default="vanilla"))
    if bypass != (loss_mode == "bypass_mode"):
        raise ValueError(
            "algorithm.rollout_correction.bypass_mode and actor_rollout_ref.actor.policy_loss.loss_mode "
            f"disagree ({bypass} vs {loss_mode!r}): set bypass_mode=true with loss_mode=bypass_mode, "
            "or bypass_mode=false with a non-bypass loss_mode"
        )
    if bypass and not bool(OmegaConf.select(config, "actor_rollout_ref.rollout.calculate_log_probs", default=False)):
        raise ValueError("bypass_mode needs actor_rollout_ref.rollout.calculate_log_probs=true (rollout log-probs)")


def check_algorithm_config(config) -> None:
    """Fail fast on combinations of the added algorithm features that train on wrong numbers."""
    from omegaconf import OmegaConf

    def sel(key, default=None):
        return OmegaConf.select(config, key, default=default)

    sentinel = sel("algorithm.invalid_reward_value") is not None
    length = bool((sel("algorithm.length_penalty") or {}).get("enable", (sel("algorithm.length_penalty") or {}).get("enabled", False)))
    gar_on = bool((sel("algorithm.gar") or {}).get("enable", False))
    arvo = bool(sel("algorithm.arvo_penalties.enable", False))
    if sentinel and not bool(sel("algorithm.exclude_invalid_rows", True)):
        raise ValueError(
            "algorithm.invalid_reward_value needs algorithm.exclude_invalid_rows=true: upstream never "
            "removes sentinel rows from the GRPO baseline (compute_advantage does not pass the config "
            "to compute_grpo_outcome_advantage), so the sentinel score would train as a reward"
        )
    if bool(sel("algorithm.use_kl_in_reward", False)) and (sentinel or length or gar_on):
        raise ValueError(
            "algorithm.use_kl_in_reward folds per-token KL into the rewards that the sentinel check, "
            "the length penalty and GAR read as outcome scores; disable it or those features"
        )
    if arvo and length:
        raise ValueError("algorithm.length_penalty and algorithm.arvo_penalties both shape length; pick one")
    if arvo and bool(sel("algorithm.tool_call_error_penalty.enable", False)) and str(
        sel("algorithm.tool_call_error_penalty.strategy", "monitor")
    ) != "monitor":
        raise ValueError(
            "algorithm.arvo_penalties and algorithm.tool_call_error_penalty both rebalance tool-error "
            "tokens (flagged negatives would get kappa twice); pick one"
        )

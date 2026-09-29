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
"""Correctness of the invalid-row / group-size / penalty fixes (verl/trainer/ppo/advantage_fixes.py).

Spec: under prompt-mean the loss is (1/|Q|) sum_q (1/T_q) sum_{i in V_q} sum_t l_{i,t}; invalid
(infra) rows belong to none of Q, V_q, T_q or the GRPO baseline.
"""

import numpy as np
import pytest
import torch

from verl.trainer.ppo import advantage_fixes as af
from verl.trainer.ppo import core_algos
from verl.utils.length_penalty import LengthPenaltyConfig


def _rewards(scores, width=4, lengths=None):
    """Outcome reward on the last valid token of each row."""
    lengths = lengths or [width] * len(scores)
    tlr = torch.zeros(len(scores), width)
    mask = torch.zeros(len(scores), width)
    for i, (s, n) in enumerate(zip(scores, lengths, strict=True)):
        mask[i, :n] = 1
        tlr[i, n - 1] = s
    return tlr, mask


def _grpo(tlr, mask, uids):
    adv, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr, response_mask=mask, index=uids, norm_adv_by_std_in_grpo=False
    )
    return adv


# --- invalid rows -------------------------------------------------------------------------


def test_invalid_rows_from_is_infra_and_sentinel():
    scores = torch.tensor([1.0, -999.0, 0.0, 0.0])
    extra = [{"is_infra": 0.0}, {}, {"is_infra": 1.0}, {"is_infra": 0.0}]
    inv = af.invalid_rows(extra, scores, -999)
    assert inv.tolist() == [False, True, True, False]
    assert af.invalid_rows(None, scores, None).tolist() == [False] * 4


def test_invalid_row_gets_zero_advantage_and_baseline_is_valid_mean():
    # group of 4: valid scores 1, 0, 1; row 3 is infra with a placeholder 0 score
    uids = np.array(["a"] * 4, dtype=object)
    tlr, mask = _rewards([1.0, 0.0, 1.0, 0.0])
    inv = torch.tensor([False, False, False, True])
    keys = [f"a_{i}_0" for i in range(4)]
    tlr = af.fill_invalid_scores(tlr, mask, uids, inv, keys)
    adv = _grpo(tlr, mask, uids)
    mean_valid = 2 / 3
    assert adv[3, -1].item() == pytest.approx(0.0, abs=1e-6)
    assert adv[0, -1].item() == pytest.approx(1 - mean_valid)
    assert adv[1, -1].item() == pytest.approx(0 - mean_valid)


def test_fill_uses_session_mean_when_sessions_have_several_rows():
    # trajectory_selection=all: session A has 3 output rows (score 1), B has 1 row (score 0),
    # C is invalid. GRPO's baseline is over one final row per session: (1 + 0 + fill) / 3.
    # The fill must be the valid *session* mean 0.5 (a row mean would give 0.75 and bias every
    # valid advantage).
    import sys
    import types

    # verl.trainer.ppo.v1 imports TransferQueue at package import; the function under test
    # does not use it.
    if "transfer_queue" not in sys.modules:
        stub = types.ModuleType("transfer_queue")
        stub.__getattr__ = lambda name: type(name, (), {})  # names only needed at import time
        sys.modules["transfer_queue"] = stub
    from verl.protocol import DataProto
    from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

    keys = ["u_A_0", "u_A_1", "u_A_2", "u_B_0", "u_C_0"]
    uids = np.array(["u"] * 5, dtype=object)
    tlr, mask = _rewards([1.0, 1.0, 1.0, 0.0, 0.0])
    inv = torch.tensor([False, False, False, False, True])
    tlr = af.fill_invalid_scores(tlr, mask, uids, inv, keys)
    assert tlr[4].sum().item() == pytest.approx(0.5)
    data = DataProto.from_dict(
        tensors={"token_level_rewards": tlr, "response_mask": mask},
        non_tensors={"uid": uids},
    )
    out = compute_advantage_for_multi_trajectories(
        data, batch_keys=keys, adv_estimator="grpo", num_repeat=3, norm_adv_by_std_in_grpo=False
    )
    adv = out.batch["advantages"][:, -1]
    assert adv[4].item() == pytest.approx(0.0, abs=1e-6)
    assert adv[0].item() == pytest.approx(0.5) and adv[3].item() == pytest.approx(-0.5)


def test_masked_rows_leave_prompt_mean_weights_untouched():
    # Prompt a: 2 valid rows + 1 invalid; prompt b: 2 valid rows. Weights of valid rows must be
    # exactly what they are without the invalid row (it counts neither tokens nor a prompt).
    uids = np.array(["a", "a", "a", "b", "b"], dtype=object)
    _, mask = _rewards([0, 0, 0, 0, 0], width=6, lengths=[3, 3, 6, 2, 4])
    inv = torch.tensor([False, False, True, False, False])
    adv = torch.ones_like(mask)
    _, masked, edited = af.mask_invalid_rows(adv, mask, inv)
    assert edited
    w = core_algos.compute_prompt_loss_weights(masked, uids)
    w_ref = core_algos.compute_prompt_loss_weights(mask[[0, 1, 3, 4]], uids[[0, 1, 3, 4]])
    assert w[2].item() == 0.0
    assert torch.allclose(w[[0, 1, 3, 4]], w_ref)
    # The old uid-reassignment path made the invalid row a third prompt: every valid weight x2/3.
    reassigned = uids.copy()
    reassigned[2] = "__infra_excluded__x"
    w_old = core_algos.compute_prompt_loss_weights(mask, reassigned)
    assert torch.allclose(w_old[[0, 1, 3, 4]], w_ref * 2 / 3)


def test_all_invalid_batch_keeps_mask_but_zero_advantage():
    adv = torch.ones(2, 3)
    mask = torch.ones(2, 3)
    out_adv, out_mask, edited = af.mask_invalid_rows(adv, mask, torch.tensor([True, True]))
    assert not edited and torch.equal(out_mask, mask) and float(out_adv.abs().sum()) == 0.0


# --- group-size correction ---------------------------------------------------------------


def test_group_size_correction_restores_common_factor():
    n_ref = 16
    uids = np.array(["full"] * 16 + ["short"] * 16, dtype=object)
    skeys = [f"{u}_{i}" for i, u in enumerate(uids)]
    inv = torch.zeros(32, dtype=torch.bool)
    inv[16 + 13 :] = True  # 3 invalid rows in the second group -> n_valid = 13
    adv = torch.ones(32, 2)
    out, m = af.group_size_correction(adv, uids, skeys, inv, n_ref=n_ref)
    assert torch.allclose(out[:16], adv[:16])  # full group untouched
    expected = (1 - 1 / 16) / (1 - 1 / 13)
    assert out[16, 0].item() == pytest.approx(expected)
    assert m["training/group_size/corrected_groups"] == 1.0


def test_group_size_correction_makes_expected_gradient_factor_common():
    # Bernoulli bandit, score function s_i ~ indicator: E[(r_i - mean) * r_i] estimates
    # (1 - 1/n) * Var(r). After correction every n gives the n_ref factor.
    rng = np.random.default_rng(0)
    p, n_ref, trials = 0.4, 16, 20000
    for n in (4, 9, 16):
        r = rng.binomial(1, p, size=(trials, n)).astype(float)
        a = r - r.mean(axis=1, keepdims=True)
        raw = (a * r).mean()
        corrected = raw * (1 - 1 / n_ref) / (1 - 1 / n)
        assert raw == pytest.approx((1 - 1 / n) * p * (1 - p), rel=0.03)
        assert corrected == pytest.approx((1 - 1 / n_ref) * p * (1 - p), rel=0.03)


def test_group_size_counts_sessions_not_rows():
    # one session with two output rows (sub-agent) counts once
    uids = np.array(["a", "a", "a"], dtype=object)
    skeys = ["a_0", "a_0", "a_1"]
    out, _ = af.group_size_correction(torch.ones(3, 1), uids, skeys, torch.zeros(3, dtype=torch.bool), n_ref=4)
    assert out[0, 0].item() == pytest.approx((1 - 1 / 4) / (1 - 1 / 2))


# --- length penalty ----------------------------------------------------------------------


def _ref_cfg():
    return LengthPenaltyConfig(
        enabled=True,
        max_penalty=0.2,
        excess_threshold=0.0,
        excess_saturate=1.0,
        penalty_exponent=1.5,
        metrics=("turns", "input_tokens", "output_tokens"),
        combine="max",
        pass_threshold=0.5,
        anchor_quantile=0.3,
        min_pass_rate=0.5,
    )


def test_length_penalty_only_long_successes_valid_sessions():
    uids = np.array(["a"] * 4, dtype=object)
    skeys = ["a_0", "a_1", "a_2", "a_3"]
    tlr, mask = _rewards([1.0, 1.0, 1.0, 0.0])
    signals = {
        "a_0": {"turn_count": 10, "prefill_length": 100, "decode_length": 100},
        "a_1": {"turn_count": 10, "prefill_length": 100, "decode_length": 100},
        "a_2": {"turn_count": 30, "prefill_length": 100, "decode_length": 100},  # 3x turns
        "a_3": {"turn_count": 90, "prefill_length": 900, "decode_length": 900},  # failure: untouched
    }
    out, m = af.apply_group_length_penalty(tlr, mask, uids, skeys, torch.zeros(4, dtype=torch.bool), signals, _ref_cfg())
    r = out.sum(dim=-1)
    assert r[0].item() == 1.0 and r[1].item() == 1.0 and r[3].item() == 0.0
    assert r[2].item() == pytest.approx(1.0 - 0.2)  # excess 2.0 saturates (s = 1)
    assert m["length_penalty/penalized_rollouts"] == 1.0


def test_length_penalty_skips_invalid_sessions_and_low_pass_groups():
    uids = np.array(["a"] * 4, dtype=object)
    skeys = ["a_0", "a_1", "a_2", "a_3"]
    tlr, mask = _rewards([1.0, 1.0, 0.0, 0.0])
    sig = {k: {"turn_count": 1 + 10 * i, "prefill_length": 1, "decode_length": 1} for i, k in enumerate(skeys)}
    inv = torch.tensor([False, False, False, True])
    out, _ = af.apply_group_length_penalty(tlr, mask, uids, skeys, inv, sig, _ref_cfg())
    # valid pass rate 2/3 > 0.5: row 1 (11 turns vs p30 anchor) is penalized, invalid row untouched
    assert out.sum(dim=-1)[1].item() < 1.0
    tlr2, mask2 = _rewards([1.0, 0.0, 0.0, 0.0])
    out2, _ = af.apply_group_length_penalty(tlr2, mask2, uids, skeys, torch.zeros(4, dtype=torch.bool), sig, _ref_cfg())
    assert torch.equal(out2, tlr2)  # pass rate 1/4 <= 0.5: no shaping


def test_session_length_signals_tensor_fallback():
    _, mask = _rewards([0, 0], width=8, lengths=[3, 5])
    sig = af.session_length_signals(None, mask, [10, 10], [8, 8], [2, 4], ["u_0", "u_1"])
    assert sig["u_0"] == {"turn_count": 2.0, "prefill_length": 15.0, "decode_length": 3.0}
    assert sig["u_1"]["prefill_length"] == 13.0


def test_legacy_length_penalty_keys():
    cfg = af.length_penalty_config({"enable": True, "deadzone": 0.3, "saturate": 1.0, "max_penalty": 0.1})
    assert cfg.enabled and cfg.excess_threshold == 0.3 and cfg.excess_saturate == 1.0
    assert af.length_penalty_config({"enable": False}) is None
    with pytest.raises(ValueError):
        af.length_penalty_config({"enable": True, "enabled": True})


# --- tool-error spans --------------------------------------------------------------------


def test_tool_error_hits_mark_whole_turn_and_fail_closed():
    _, mask = _rewards([0, 0, 0], width=10)
    extra = [
        {"llm_turn_spans": [[0, 3], [5, 8]], "tool_call_error_flags": [False, True]},
        {"llm_turn_spans": [[0, 3]], "tool_call_error_flags": [True, False]},  # misaligned
        {"llm_turn_spans": [[0, 4], [2, 6]], "tool_call_error_flags": [True, True]},  # overlapping
    ]
    hits, m = af.tool_error_hits_from_spans(extra, mask)
    assert hits[0].tolist() == [False] * 5 + [True] * 3 + [False] * 2
    assert not hits[1].any() and not hits[2].any()
    assert m["penalty/tool_call_error_span_misaligned_rows"] == 2.0


# --- arvo reference path (previously raised TypeError on every call) ---------------------


def test_reference_apply_tool_penalty_runs_and_conserves_prompt_mean_mass():
    from verl.trainer.ppo.arvo_penalties import ReferencePenalties

    ref = ReferencePenalties.reference()
    adv = torch.tensor([[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5]])
    mask = torch.ones_like(adv)
    infos = [
        {"llm_turn_spans": [[0, 1], [1, 4]], "tool_call_error_flags": [True, False]},
        {"llm_turn_spans": [[0, 3], [3, 4]], "tool_call_error_flags": [False, True]},
    ]
    uids = np.array(["a", "a"], dtype=object)
    w = core_algos.compute_prompt_loss_weights(mask, uids)
    out, m = ref.apply_tool_penalty(adv, mask, mask.clone(), infos, row_weights=w)
    assert out[0, 0].item() == 0.0 and out[0, 1].item() > 0.5  # flagged turn zeroed, rest scaled up
    assert out[1, 3].item() == pytest.approx(-0.5 * ref.kappa)
    assert m["penalty/signed/neg_scale_clamped"] == 0 and m["penalty/signed/pos_scale_clamped"] == 0
    assert m["penalty/signed/adv_pos_sum_post"] == pytest.approx(m["penalty/signed/adv_pos_sum_pre"])
    assert m["penalty/signed/adv_neg_sum_post"] == pytest.approx(m["penalty/signed/adv_neg_sum_pre"])


def test_trajectory_metadata_clips_overhanging_spans():
    from recipes.general.trajectory_metadata import trajectory_metadata

    meta = trajectory_metadata([1, 2], [1, 1, 0, 1, 1], [(0, 2), (3, 7), (9, 12)], [False, True, True])
    assert meta["llm_turn_spans"] == [[0, 2], [3, 5]]
    assert meta["tool_call_error_flags"] == [False, True]


def test_exec_budget_hit_reads_both_recipe_layouts():
    assert af.exec_budget_hit({"exec_budget_hit": 1.0}) == 1.0  # general: extra_fields
    assert af.exec_budget_hit({"reward_extra_info": {"exec_budget_hit": True}}) == 1.0  # code: reward_info
    assert af.exec_budget_hit({"reward_extra_info": {"exec_budget_hit": False}}) == 0.0
    assert af.exec_budget_hit({}) == 0.0

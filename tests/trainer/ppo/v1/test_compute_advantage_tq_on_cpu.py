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
"""``PPOTrainer._compute_advantage`` end to end through a real TransferQueue partition.

The unit tests in ``tests/trainer/ppo/test_advantage_fixes_on_cpu.py`` cover each helper; this
one checks that the trainer wires them together: rows are read from and written back to
TransferQueue as jagged tensors, invalid rows leave the baseline and the loss, a multi-row
session is graded on its final row, the length penalty and the group-size correction apply in
the right order, and the prompt-mean weights see the edited mask.
"""

import uuid

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

tq = pytest.importorskip("transfer_queue")

from verl.trainer.ppo.v1.trainer_base import PPOTrainer  # noqa: E402
from verl.utils.tensordict_utils import list_of_dict_to_tensordict  # noqa: E402

G = 4


@pytest.fixture(scope="module", autouse=True)
def tq_init():
    tq.init()
    yield
    tq.close()


class _Trainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(**algo_overrides):
    algorithm = {
        "adv_estimator": "grpo",
        "norm_adv_by_std_in_grpo": False,
        "gamma": 1.0,
        "lam": 1.0,
        "use_kl_in_reward": False,
        "invalid_reward_value": None,
        "exclude_invalid_rows": True,
        "group_size_correction": True,
        "length_penalty": None,
        "tool_call_error_penalty": {"enable": False},
        "rollout_correction": {"bypass_mode": True},
    }
    algorithm.update(algo_overrides)
    config = OmegaConf.create(
        {
            "algorithm": algorithm,
            "actor_rollout_ref": {"rollout": {"n": G}, "actor": {"loss_agg_mode": "prompt-mean"}},
            "critic": {"ppo_mini_batch_size": 1},
        }
    )
    trainer = object.__new__(_Trainer)
    trainer.config = config
    trainer.reference_penalties = None
    return trainer


def _row(key, score, length, *, infra=False, signals=None, spans=None, flags=None):
    uid = key.split("_")[0]
    rm = torch.zeros(length)
    rm[-1] = score
    extra = {"is_infra": 1.0 if infra else 0.0}
    if signals is not None:
        extra["length_signals"] = dict(zip(("turn_count", "prefill_length", "decode_length"), signals, strict=True))
    if spans is not None:
        extra["llm_turn_spans"] = spans
        extra["tool_call_error_flags"] = flags
    fields = {
        "uid": uid,
        "response_mask": torch.ones(length, dtype=torch.int64),
        "rm_scores": rm,
        "rollout_log_probs": torch.zeros(length),
        "old_log_probs": torch.zeros(length),
        "extra_fields": extra,
    }
    tag = {"prompt_len": 5, "response_len": length}
    return key, fields, tag


def _run(trainer, rows):
    partition = f"adv-{uuid.uuid4().hex}"
    keys = [r[0] for r in rows]
    meta = tq.kv_batch_put(
        keys=keys,
        partition_id=partition,
        fields=list_of_dict_to_tensordict([r[1] for r in rows]),
        tags=[r[2] for r in rows],
    )
    metrics = {}
    trainer._compute_advantage(meta, metrics)
    out = tq.kv_batch_get(
        keys=keys,
        partition_id=partition,
        select_fields=["advantages", "returns", "response_mask", "prompt_loss_weights"],
    )
    res = {}
    for i, k in enumerate(keys):
        res[k] = {
            "adv": out["advantages"][i].clone(),
            "ret": out["returns"][i].clone(),
            "mask": out["response_mask"][i].clone(),
            "w": float(out["prompt_loss_weights"][i]),
        }
    return res, metrics


def _length_penalty(ls, anchor, max_penalty=0.2, exponent=1.5):
    excess = max(v / a - 1.0 for v, a in zip(ls, anchor, strict=True))
    t = min(max(excess, 0.0), 1.0)
    return max_penalty * t**exponent


LP = {
    "enable": True,
    "max_penalty": 0.2,
    "excess_threshold": 0.0,
    "excess_saturate": 1.0,
    "penalty_exponent": 1.5,
    "metrics": ["turns", "input_tokens", "output_tokens"],
    "combine": "max",
    "pass_threshold": 0.5,
    "anchor_quantile": 0.3,
    "min_pass_rate": 0.5,
}


def _scenario_rows():
    # prompt a: session 0 has two output rows (graded on the last), session 2 is an infra
    # failure, session 3 is a long success. prompt b: one success out of four.
    return [
        _row("a_0_0", 0.0, 3, signals=(1, 5, 3)),
        _row("a_0_1", 1.0, 3, signals=(1, 5, 3)),
        _row("a_1_0", 0.0, 4, signals=(3, 12, 4)),
        _row("a_2_0", 0.0, 2, infra=True, signals=(0, 0, 0)),
        _row("a_3_0", 1.0, 12, signals=(4, 20, 12)),
        _row("b_0_0", 1.0, 5, signals=(2, 10, 5)),
        _row("b_1_0", 0.0, 5, signals=(2, 10, 5)),
        _row("b_2_0", 0.0, 5, signals=(2, 10, 5)),
        _row("b_3_0", 0.0, 5, signals=(2, 10, 5)),
    ]


def test_invalid_rows_length_penalty_and_group_size_correction():
    res, metrics = _run(_trainer(length_penalty=LP), _scenario_rows())

    # Length penalty on prompt a (valid pass rate 2/3 > 0.5): session signals are summed over
    # its rows, the anchor is the p30 of the successful sessions.
    s0 = (2, 10, 6)
    s3 = (4, 20, 12)
    anchor = [float(np.quantile([x, y], 0.3)) for x, y in zip(s0, s3, strict=True)]
    r_a = {0: 1.0 - _length_penalty(s0, anchor), 1: 0.0, 3: 1.0 - _length_penalty(s3, anchor)}
    assert r_a[3] < 1.0 and r_a[0] == pytest.approx(1.0)
    mean_a = sum(r_a.values()) / 3
    corr_a = (1 - 1 / G) / (1 - 1 / 3)
    expected_a = {s: (r - mean_a) * corr_a for s, r in r_a.items()}

    for key, sess in (("a_0_0", 0), ("a_0_1", 0), ("a_1_0", 1), ("a_3_0", 3)):
        assert torch.allclose(res[key]["adv"], torch.full_like(res[key]["adv"], expected_a[sess]), atol=1e-6), key
    # Advantages of the valid final rows sum to zero within the group.
    assert sum(expected_a.values()) == pytest.approx(0.0, abs=1e-7)

    # The infra row: no advantage, no loss tokens, no prompt-mean weight.
    assert torch.count_nonzero(res["a_2_0"]["adv"]) == 0
    assert torch.count_nonzero(res["a_2_0"]["mask"]) == 0
    assert res["a_2_0"]["w"] == 0.0

    # Prompt b is a full group: plain GRPO, no correction, no length penalty (pass rate 1/4).
    for key, r in (("b_0_0", 1.0), ("b_1_0", 0.0), ("b_2_0", 0.0), ("b_3_0", 0.0)):
        assert torch.allclose(res[key]["adv"], torch.full_like(res[key]["adv"], r - 0.25), atol=1e-6)

    for key in res:
        assert torch.equal(res[key]["ret"], res[key]["adv"])

    # prompt-mean: every prompt weighs 1 / (#prompts * its valid tokens).
    tokens_a = 3 + 3 + 4 + 12
    assert res["a_0_0"]["w"] == pytest.approx(1 / (2 * tokens_a))
    assert res["b_0_0"]["w"] == pytest.approx(1 / (2 * 20))
    assert metrics["training/invalid_rows"] == 1.0
    assert metrics["loss/prompt_mean_prompt_count"] == 2.0


def test_exclude_invalid_rows_off_trains_infra_rows_as_ordinary_rows():
    res, metrics = _run(_trainer(exclude_invalid_rows=False), _scenario_rows())

    # Upstream behavior: the infra row stays in the group with its 0 score and keeps its tokens.
    mean_a = (1.0 + 0.0 + 0.0 + 1.0) / 4
    for key, r in (("a_0_1", 1.0), ("a_1_0", 0.0), ("a_2_0", 0.0), ("a_3_0", 1.0)):
        assert torch.allclose(res[key]["adv"], torch.full_like(res[key]["adv"], r - mean_a), atol=1e-6), key
    assert torch.count_nonzero(res["a_2_0"]["mask"]) == 2
    assert res["a_2_0"]["w"] > 0.0
    assert metrics["training/invalid_rows"] == 0.0


def test_tool_error_spans_zero_only_the_failed_turn_of_a_success():
    rows = [
        _row("c_0_0", 1.0, 6, spans=[[0, 3], [3, 6]], flags=[False, True]),
        _row("c_1_0", 0.0, 6, spans=[[0, 3], [3, 6]], flags=[False, False]),
        _row("c_2_0", 0.0, 6, spans=[[0, 6]], flags=[False]),
        _row("c_3_0", 0.0, 6, spans=[[0, 6]], flags=[False]),
    ]
    tool = {"enable": True, "strategy": "adv_signed", "penalty_value": 2.0, "mask_source": "spans"}
    res, _ = _run(_trainer(tool_call_error_penalty=tool), rows)

    adv = res["c_0_0"]["adv"]
    assert torch.count_nonzero(adv[3:]) == 0  # the failed turn of a positive rollout
    assert bool((adv[:3] > 0.75).all())  # its removed mass moved to the clean tokens
    # No flagged token on the negative side: its scale stays 1.
    for key in ("c_1_0", "c_2_0", "c_3_0"):
        assert torch.allclose(res[key]["adv"], torch.full_like(res[key]["adv"], -0.25), atol=1e-6)

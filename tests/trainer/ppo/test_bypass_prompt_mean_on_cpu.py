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
"""Report Eq. (1) through the bypass_mode loss with prompt-mean aggregation.

L = -sum_t sg[pi/mu] * M * A * log pi, M = 0 outside [0.2, 5.0]; masked tokens stay in the
prompt-mean normalizer. A micro-batch holding only masked rows must give a zero loss and the
same metric keys as any other micro-batch.
"""

import math

import pytest
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config import FSDPActorConfig, PolicyLossConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig

RC = {"bypass_mode": True, "loss_type": "reinforce", "rollout_is": "token", "rollout_is_threshold": "0.2_5.0"}


def _actor(loss_mode="bypass_mode"):
    return FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="prompt-mean",
        policy_loss=PolicyLossConfig(loss_mode=loss_mode, rollout_correction=RC),
        optim=FSDPOptimizerConfig(lr=1e-6),
    )


def _loss(actor, log_prob, rollout_log_prob, mask, weights):
    actor.global_batch_info.clear()
    actor.global_batch_info.update({"dp_size": 1, "prompt_loss_weights": weights})
    return get_policy_loss_fn("bypass_mode")(
        old_log_prob=rollout_log_prob,
        log_prob=log_prob,
        advantages=torch.ones_like(log_prob),
        response_mask=mask,
        loss_agg_mode="prompt-mean",
        config=actor,
    )


def test_eq1_mask_and_normalizer():
    actor = _actor()
    log_prob = torch.zeros(2, 3, requires_grad=True)
    # row 0 ratios pi/mu: 1, 10 (outside [0.2, 5]), 0.5; row 1 is an invalid (masked) row
    rollout_log_prob = torch.tensor([[0.0, -math.log(10), math.log(2)], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    loss, metrics = _loss(actor, log_prob, rollout_log_prob, mask, torch.tensor([1 / 3, 0.0]))
    loss.backward()
    # d/dlogpi of -(w * A * logpi) / |o| with |o| = 3 counting the masked-out token
    assert torch.allclose(log_prob.grad[0], torch.tensor([-1 / 3, 0.0, -0.5 / 3]))
    assert torch.equal(log_prob.grad[1], torch.zeros(3))
    assert metrics["rollout_corr/rollout_is_oob_ratio"] == pytest.approx(1 / 3)


def test_all_masked_micro_batch():
    actor = _actor()
    _, ref_metrics = _loss(
        actor,
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([1 / 3]),
    )
    log_prob = torch.zeros(1, 3, requires_grad=True)
    loss, metrics = _loss(actor, log_prob, torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool), torch.tensor([0.0]))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(log_prob.grad, torch.zeros(1, 3))
    assert set(metrics) == set(ref_metrics)
    assert all(v == 0.0 for k, v in metrics.items() if k.startswith("rollout_corr/"))


def test_prompt_mean_rejects_losses_without_batch_info():
    with pytest.raises(ValueError, match="prompt-mean"):
        _actor(loss_mode="gpg")

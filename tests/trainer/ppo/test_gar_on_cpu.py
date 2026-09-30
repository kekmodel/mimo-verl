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
"""GAR redistribution (report 4.3.2, GAGAR arXiv 2609.32577 Eq. 2-11, A.1-A.3)."""

import pytest

from verl.trainer.ppo.gar import (
    Candidate,
    GARConfig,
    Grade,
    Group,
    GroupResult,
    factors,
    redistribute,
    validate_result,
)

CFG = GARConfig()


def test_factor_map_is_the_papers_flash_configuration():
    grades = {
        "a": Grade("T1", 0),
        "b": Grade("T1", 0),
        "c": Grade("T1", 1),
        "d": Grade("T2", 0),
        "e": Grade("T2", 1),
        "f": Grade("T2", 1),
        "g": Grade("T2", 4),  # tied-group order, not the raw rank value, sets the step
        "h": Grade("T3", 0),
    }
    f = factors(grades, CFG)
    assert f["a"] == f["b"] == 1.0
    assert f["c"] == 0.9
    assert f["d"] == pytest.approx(0.85)
    assert f["e"] == f["f"] == pytest.approx(0.625)
    assert f["g"] == pytest.approx(0.4)
    assert f["h"] == 0.2
    assert factors({"x": Grade("T2", 3)}, CFG)["x"] == 0.85


def test_binary_redistribution_preserves_the_positive_sum_and_the_factor_ratios():
    scores = [1.0, 1.0, 1.0] + [0.0] * 13
    passed = [True] * 3 + [False] * 13
    f = [1.0, 0.85, 0.4] + [None] * 13
    new, info = redistribute(scores, passed, f, lambda_max=1.5)
    mean = 3 / 16
    assert sum(new[:3]) == pytest.approx(3 * (1 - mean))
    assert new[1] / new[0] == pytest.approx(0.85) and new[2] / new[0] == pytest.approx(0.4)
    assert all(x == pytest.approx(-mean) for x in new[3:])
    assert sum(new) == pytest.approx(0.0, abs=1e-12)
    assert info["capped"] == 0.0
    # Reward-space equivalent (A.3): A* = (1 - R) f / mean_P(f).
    fbar = (1.0 + 0.85 + 0.4) / 3
    assert new[0] == pytest.approx((1 - mean) * 1.0 / fbar)


def test_cap_then_recentre():
    # Two passes, f = (1, 0.2): lambda = 2 / 1.2 > 1.5, so it is capped and the group
    # mean is subtracted (the dashboard's 1.32344 maximum).
    scores = [1.0, 1.0] + [0.0] * 14
    new, info = redistribute(scores, [True, True] + [False] * 14, [1.0, 0.2] + [None] * 14, lambda_max=1.5)
    assert info["capped"] == 1.0 and info["lambda"] == 1.5
    assert sum(new) == pytest.approx(0.0, abs=1e-12)
    assert max(new) == pytest.approx(1.32344, abs=1e-5)


def test_nonbinary_scores_use_the_positive_part_of_passes():
    # A length-shaped pass that lands below the group mean keeps no positive credit (Eq. 8).
    scores = [1.0, 0.1, 0.0, 0.0]
    passed = [True, True, False, False]
    new, _ = redistribute(scores, passed, [1.0, 0.85, None, None], lambda_max=1.5)
    mean = sum(scores) / 4
    a = [s - mean for s in scores]
    b = [1.0 * a[0], 0.0, a[2], a[3]]  # lambda = a0 / (1 * a0) = 1
    mb = sum(b) / 4
    assert new == pytest.approx([x - mb for x in b])


def test_equal_factors_or_single_pass_leave_grpo_unchanged():
    scores = [1.0, 1.0, 0.0, 0.0]
    new, _ = redistribute(scores, [True, True, False, False], [0.85, 0.85, None, None], lambda_max=1.5)
    assert new == pytest.approx([0.5, 0.5, -0.5, -0.5])
    new, _ = redistribute([1.0, 0.0, 0.0], [True, False, False], [0.85, None, None], lambda_max=1.5)
    assert new == pytest.approx([2 / 3, -1 / 3, -1 / 3])
    # Below 1 / lambda_max even equal factors hit the cap (paper A.2: the bounded branch does
    # not preserve the positive sum): the passes stay tied but lose credit.
    new, info = redistribute(scores, [True, True, False, False], [0.625, 0.625, None, None], lambda_max=1.5)
    assert info["capped"] == 1.0 and new[0] == new[1] < 0.5 and sum(new) == pytest.approx(0.0, abs=1e-12)


def test_unusable_results_are_rejected():
    group = Group("u", [Candidate("u_0", True, 1.0, {}), Candidate("u_1", True, 1.0, {}), Candidate("u_2", False, 0.0, {})])
    assert validate_result(group, None) is not None
    assert "missing" in validate_result(group, GroupResult({"u_0": Grade("T1", 0)}))
    assert validate_result(group, GroupResult({"u_0": Grade("T1", 0)}, hacks=["u_1"])) is None
    assert validate_result(group, GroupResult({"u_0": Grade("T4", 0), "u_1": Grade("T1", 0)})) is not None
    assert validate_result(group, GroupResult({"u_0": Grade("T1", 0), "u_1": Grade("T1", 0)}, hacks=["u_2"])) is not None


def test_config_validation():
    assert GARConfig.from_raw(None) is None
    assert GARConfig.from_raw({"enable": False}) is None
    with pytest.raises(ValueError, match="grader"):
        GARConfig(enable=True)
    with pytest.raises(ValueError, match="f_min"):
        GARConfig(f_min=0.9, f_max=0.5)
    with pytest.raises(ValueError, match="lambda_max"):
        GARConfig(lambda_max=0.5)

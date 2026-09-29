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
"""Runner-owned tool-execution budget (recipes/code/mimoagent_runner._ExecBudget).

The budget counts only pod-command time (the policy's), never model-query time (which also
contains training pauses and inference queueing). Once used up, the rollout is frozen -- no
further model query or pod command -- so the final state can be graded without racing the
agent; the liveness probe bypasses the freeze.
"""

import time

import pytest

pytest.importorskip("mimoagent")
from mimoagent.agents.base import LimitsExceeded  # noqa: E402

from recipes.code.mimoagent_runner import _ExecBudget  # noqa: E402


class _Model:
    def query(self, messages, **kwargs):
        time.sleep(0.2)  # slow generation / training pause: must not count
        return {"content": "ok"}


class _Env:
    def __init__(self):
        self.commands = []

    def execute(self, command, **kwargs):
        self.commands.append(command)
        time.sleep(0.05)
        return {"output": "", "returncode": 0, "reason": "ok"}


def test_model_time_does_not_count_tool_time_does():
    model, env = _Model(), _Env()
    budget = _ExecBudget(0.12)
    budget.install(model, env)
    for _ in range(3):
        model.query([])  # 0.6s of generation, budget untouched
    assert budget.used == 0.0 and not budget.hit
    env.execute("a")
    env.execute("b")  # 0.10s used, still under budget
    env.execute("c")  # crosses the budget; this command still completes
    assert env.commands == ["a", "b", "c"] and budget.used >= 0.12
    with pytest.raises(LimitsExceeded):
        env.execute("d")
    with pytest.raises(LimitsExceeded):
        model.query([])
    assert budget.hit and env.commands == ["a", "b", "c"]  # frozen command never reached the pod
    assert budget.env_alive() and env.commands[-1] == "true"  # probe bypasses the freeze

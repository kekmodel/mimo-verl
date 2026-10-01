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
"""One uni-agent runner for several task sources (Code + General in one run).

uni-agent picks a runner by the row's ``agent_name``, and every MimoAgent dataset ships
``agent_name: mimo_swe_agent``. Instead of rewriting the data, this single runner routes each
sample by its instance's ``dataset_type`` to a named route and runs the MimoAgent runner with the
shared kwargs overlaid by the route's::

    runner_kwargs:
      route_by_dataset_type: {opensource-code: code, general_agent: general, terminal_bench: general}
      routes:
        code: {exec_budget_seconds: 3000}
        general: {config_path: config/agent/general/s3k-uni.yaml, environment_hooks: general,
                  reward_binarize_threshold: 1.0, exec_budget_seconds: 300}
      exec_budget_probe_timeout: 30        # shared by every route unless a route overrides it

The route name is written to ``reward_info["source"]``. A dataset_type with no route fails the
session loudly rather than running it under another source's settings.
"""

from __future__ import annotations

from typing import Any

from recipes.code.mimoagent_runner import mimoagent_runner


def resolve_route(
    tools_kwargs: dict | None, route_by_dataset_type: dict[str, str], routes: dict[str, dict]
) -> tuple[str, dict[str, Any]]:
    instance = (tools_kwargs or {}).get("instance")
    if not isinstance(instance, dict):
        raise ValueError("mixed runner needs tools_kwargs.instance")
    dataset_type = instance.get("dataset_type")
    name = route_by_dataset_type.get(str(dataset_type))
    if name is None:
        raise ValueError(f"no route for dataset_type {dataset_type!r}; known: {sorted(route_by_dataset_type)}")
    if name not in routes:
        raise ValueError(f"route {name!r} (dataset_type {dataset_type!r}) is not defined in routes")
    return name, dict(routes[name] or {})


async def mixed_runner(
    *,
    raw_prompt,
    session,
    sample_index: int,
    tools_kwargs: dict | None = None,
    routes: dict[str, dict],
    route_by_dataset_type: dict[str, str],
    **shared,
) -> None:
    name, route = resolve_route(tools_kwargs, route_by_dataset_type, routes)
    kwargs = {**shared, **route, "source": name}
    await mimoagent_runner(raw_prompt=raw_prompt, session=session, sample_index=sample_index, tools_kwargs=tools_kwargs, **kwargs)

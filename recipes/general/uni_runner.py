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
"""General tasks (general_agent / terminal_bench) on uni-agent: the environment hooks the
MimoAgent runner (``recipes/code/mimoagent_runner.py``) calls when a route sets
``environment_hooks: general``.

``GeneralAgentLoop`` + ``DatasetEnvActor`` do the same work for verl's own AgentLoop; this
module carries the parts that are not in mimoagent itself, so one uni-agent run can serve both
Code and General tasks:

* register the General dataset environment and, when the agent config names them, the
  Claude-Code-style tools (``register_cc_tools``);
* pass the cluster settings the env actor added (labels, KUBECONFIG, DOCKER_REGISTRY);
* discover the instance's MCP tools through the in-pod bridge, add them to the agent's tool
  registry, and copy the bridge into the sidecar for live-world grading.

MCP discovery and the sidecar copy run before the runner installs the tool-execution budget,
so they are not charged; MCP tool calls the agent makes go through ``env.execute`` and are.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class GeneralHooks:
    name = "general"

    def before_environment(self, instance: dict[str, Any], environment_config: dict[str, Any], config: dict) -> None:
        from mimoagent.tools.registry import _TOOL_CLASSES

        from .general_agent import register_general_agent_env

        register_general_agent_env()
        # The bundle's parquet may give env_task_dir relative to GA_TASK_ROOT (run_general.sh
        # cds there); a uni-agent runner runs elsewhere, so anchor it.
        task_dir = instance.get("env_task_dir")
        root = os.environ.get("GA_TASK_ROOT")
        if task_dir and not os.path.isabs(task_dir) and root:
            instance["env_task_dir"] = os.path.join(root, task_dir)
        tools = [t.get("tool") for t in (config.get("agent") or {}).get("tools") or [] if isinstance(t, dict)]
        if any(name not in _TOOL_CLASSES for name in tools):
            from .tools import register_cc_tools

            register_cc_tools()
        environment_config["labels"] = {
            "exp": os.getenv("EXP_NAME", "unknown"),
            **(environment_config.get("labels") or {}),
        }
        if os.environ.get("KUBECONFIG") and "kubeconfig" not in environment_config:
            environment_config["kubeconfig"] = os.environ["KUBECONFIG"]
        if os.environ.get("DOCKER_REGISTRY") and "image_prefix" not in environment_config:
            environment_config["image_prefix"] = os.environ["DOCKER_REGISTRY"]

    def after_agent(self, environment, agent, instance: dict[str, Any]) -> dict[str, Any]:
        """Add the instance's MCP tools to ``agent``. Returns facts for reward_info."""
        env = environment.env
        servers = getattr(env, "mcp_servers", None)
        bridge = getattr(env, "mcp_bridge_script", None)
        if not servers:
            return {"mcp_tools": 0}
        if not bridge:
            logger.warning("instance %s has MCP servers but no bridge script; MCP tools not registered", instance.get("instance_id"))
            return {"mcp_tools": 0, "mcp_tools_error": "no bridge script"}
        from .mcp_proxy import discover_mcp_tools

        tools = discover_mcp_tools(env, servers, getattr(env, "mcp_bridge_python", "python3"), bridge)
        for tool in tools:
            agent.tool_registry.register(tool)
        # DefaultAgent caches the definitions it sends with every model call at construction.
        if hasattr(agent, "_tool_definitions"):
            agent._tool_definitions = agent.tool_registry.get_function_definitions()
        self._copy_bridge_to_sidecar(env, instance, bridge)
        return {"mcp_tools": len(tools)}

    @staticmethod
    def _copy_bridge_to_sidecar(env, instance: dict[str, Any], bridge_path: str) -> None:
        """Live-world graders run in the sidecar and read the bridge there (see
        ``DatasetEnvActor._ensure_bridge_in_sidecar``). Best effort."""
        src = os.path.join(instance.get("env_task_dir") or "", "mcp_bridge.py")
        if not os.path.isfile(src):
            return
        try:
            env.execute(f"mkdir -p {os.path.dirname(bridge_path)}", container="sidecar")
            env.copy_to(src, bridge_path, container="sidecar", dereference=True)
        except Exception as e:  # noqa: BLE001 - leaves the degraded scoring path, never fails the rollout
            logger.warning("could not copy the MCP bridge into the sidecar: %s", e)


HOOKS = {"general": GeneralHooks}


def get_hooks(name: str | None):
    if not name:
        return None
    if name not in HOOKS:
        raise ValueError(f"unknown environment_hooks {name!r}; known: {sorted(HOOKS)}")
    return HOOKS[name]()

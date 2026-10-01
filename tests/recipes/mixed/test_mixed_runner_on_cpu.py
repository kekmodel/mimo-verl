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
"""Code + General on one uni-agent runner: routing, General hooks, the mixed config."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mimoagent")

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTES = {
    "code": {"exec_budget_seconds": 3000},
    "general": {"config_path": "config/agent/general/s3k-uni.yaml", "environment_hooks": "general", "exec_budget_seconds": 300},
}
BY_TYPE = {"opensource-code": "code", "general_agent": "general", "terminal_bench": "general"}


def test_routes_by_dataset_type_and_refuses_unknown():
    from recipes.mixed.runner import resolve_route

    name, route = resolve_route({"instance": {"dataset_type": "general_agent"}}, BY_TYPE, ROUTES)
    assert name == "general" and route["environment_hooks"] == "general"
    with pytest.raises(ValueError, match="no route"):
        resolve_route({"instance": {"dataset_type": "webdev"}}, BY_TYPE, ROUTES)
    with pytest.raises(ValueError, match="not defined"):
        resolve_route({"instance": {"dataset_type": "general_agent"}}, BY_TYPE, {"code": {}})


def test_route_kwargs_overlay_the_shared_ones(monkeypatch):
    import recipes.mixed.runner as mixed

    seen = {}

    async def fake_runner(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(mixed, "mimoagent_runner", fake_runner)
    asyncio.run(
        mixed.mixed_runner(
            raw_prompt="x",
            session=None,
            sample_index=0,
            tools_kwargs={"instance": {"dataset_type": "general_agent"}},
            routes=ROUTES,
            route_by_dataset_type=BY_TYPE,
            exec_budget_seconds=3000,
            exec_budget_probe_timeout=30,
        )
    )
    assert seen["exec_budget_seconds"] == 300 and seen["exec_budget_probe_timeout"] == 30
    assert seen["source"] == "general" and seen["config_path"].endswith("s3k-uni.yaml")


class _Registry:
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def get_function_definitions(self):
        return [{"name": n} for n in self.tools]


def test_general_hooks_register_mcp_tools_and_binarize(monkeypatch, tmp_path):
    from recipes.code import mimoagent_runner as runner
    from recipes.general import uni_runner

    (tmp_path / "envs" / "t1").mkdir(parents=True)
    monkeypatch.setenv("GA_TASK_ROOT", str(tmp_path))
    seen_instance = {}

    class Env:
        mcp_servers = {"crm": {}}
        mcp_bridge_script = "/work/_setup/mcp_bridge.py"

        def execute(self, command, **kwargs):
            return {"output": "", "returncode": 0}

        def copy_to(self, *a, **k):
            pass

    class FakeEnvironment:
        env = Env()

        def setup_environment(self):
            pass

        def calculate_reward(self):
            return 0.7, "rubric 7/10", {"rubric_reward": 0.7}

        def cleanup(self):
            pass

    class FakeAgent:
        IDLE_STATUS = "Idle"

        def __init__(self, model, env, **kwargs):
            self.tool_registry = _Registry()
            self._tool_definitions = []

        def run(self, task):
            return "Idle", "done"

    def fake_make_env(instance, **kwargs):
        seen_instance.update(instance)
        seen_instance["_env_kwargs"] = kwargs
        return FakeEnvironment()

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: FakeAgent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", fake_make_env)
    monkeypatch.setattr(uni_runner.GeneralHooks, "before_environment", _before_without_registration(uni_runner))
    monkeypatch.setattr(
        "recipes.general.mcp_proxy.discover_mcp_tools", lambda env, servers, py, bridge: [SimpleNamespace(name="crm.find")]
    )
    monkeypatch.setenv("KUBECONFIG", "/k/config")

    config = runner._load_config(REPO_ROOT / "config/agent/general/s3k-uni.yaml")
    config["agent"] = {"type": "default"}
    result = runner._run_sync(
        raw_prompt="do it",
        instance={"dataset_type": "general_agent", "env_task_dir": "envs/t1", "problem_statement": "do it"},
        session=SimpleNamespace(base_url="http://gateway/s/v1"),
        config=config,
        agent_overrides={},
        environment_overrides={},
        environment_hooks=uni_runner.get_hooks("general"),
        reward_binarize_threshold=1.0,
    )
    assert seen_instance["env_task_dir"] == os.path.join(str(tmp_path), "envs/t1")
    assert seen_instance["_env_kwargs"]["kubeconfig"] == "/k/config"
    assert result["mcp_tools"] == 1
    assert result["raw_reward"] == 0.7 and result["reward"] == 0.0  # 0.7 < 1.0: fail


def _before_without_registration(uni_runner):
    """before_environment minus the mimoagent registry calls (no General env package in CI)."""
    original = uni_runner.GeneralHooks.before_environment

    def before(self, instance, environment_config, config):
        import recipes.general.general_agent as ga

        saved = ga.register_general_agent_env
        ga.register_general_agent_env = lambda: None
        try:
            return original(self, instance, environment_config, {"agent": {"tools": []}})
        finally:
            ga.register_general_agent_env = saved

    return before


def test_mcp_tools_reach_the_real_cc_agent(monkeypatch):
    """With the real cc-agent built from s3k-uni.yaml: the MCP tools must be in the definitions
    sent with every model call and executable by name."""
    from mimoagent.agents.factory import get_agent_class

    from recipes.code import mimoagent_runner as runner
    from recipes.general import uni_runner
    from recipes.general.mcp_proxy import McpProxyTool

    config = runner._load_config(REPO_ROOT / "config/agent/general/s3k-uni.yaml")
    agent_config = dict(config["agent"])
    agent_cls = get_agent_class(agent_config.pop("type"))
    model = SimpleNamespace(query=lambda *a, **k: {"content": ""}, get_template_vars=lambda: {}, config=SimpleNamespace(model_name="policy"))
    env = SimpleNamespace(
        mcp_servers={"crm": {}},
        mcp_bridge_script="/b.py",
        execute=lambda *a, **k: {"output": "", "returncode": 0},
        get_template_vars=lambda: {"cwd": "/work/workspace"},
        config=SimpleNamespace(cwd="/work/workspace"),
    )
    agent = agent_cls(model, env, **agent_config)
    before = {d["function"]["name"] if "function" in d else d.get("name") for d in agent._tool_definitions}
    tool = McpProxyTool(
        server="crm", fn="find", description="find a record", input_schema={"type": "object", "properties": {}},
        url="http://crm", bridge_python="python3", bridge_script="/b.py",
    )
    monkeypatch.setattr("recipes.general.mcp_proxy.discover_mcp_tools", lambda *a, **k: [tool])
    info = uni_runner.GeneralHooks().after_agent(SimpleNamespace(env=env), agent, {})
    names = {d["function"]["name"] if "function" in d else d.get("name") for d in agent._tool_definitions}
    assert info == {"mcp_tools": 1}
    assert names - before == {tool.name}
    assert agent.tool_registry.get(tool.name) is tool


def test_relative_config_path_resolves_from_the_repo(monkeypatch):
    from recipes.code import mimoagent_runner as runner

    seen = {}

    def fake_run_sync(**kwargs):
        seen.update(kwargs)
        return {"reward": 1.0, "agent_type": "cc-agent", "agent_status": "Idle"}

    async def no_post(*a, **k):
        return None

    monkeypatch.setattr(runner, "_run_sync", fake_run_sync)
    monkeypatch.setattr(runner, "_post_reward_info_with_retry", no_post)
    asyncio.run(
        runner.mimoagent_runner(
            raw_prompt="x",
            session=SimpleNamespace(base_url="http://g/v1", reward_info_url="http://g/r"),
            sample_index=0,
            tools_kwargs={"instance": {"dataset_type": "general_agent"}},
            config_path="config/agent/general/s3k-uni.yaml",
            source="general",
        )
    )
    assert seen["config"]["agent"]["type"] == "cc-agent" and seen["config"]["model"]["model_name"] == "policy"


def test_mixed_config_composes():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from verl.trainer.ppo import advantage_fixes

    with initialize_config_dir(config_dir=str(REPO_ROOT / "recipes/mixed/config"), version_base=None):
        cfg = compose(
            config_name="mixed",
            overrides=[f"hydra.searchpath=[pkg://verl.trainer.config,file://{REPO_ROOT}/recipes/code/config]"],
        )
    OmegaConf.resolve(cfg)
    runner_cfg = cfg.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mimoagent
    assert runner_cfg.runner_fqn == "recipes.mixed.runner.mixed_runner"
    kwargs = runner_cfg.runner_kwargs
    assert kwargs.route_by_dataset_type["general_agent"] == "general"
    assert kwargs.routes.general.reward_binarize_threshold == 1.0
    assert kwargs.exec_budget_seconds  # shared Code setting still inherited
    assert runner_cfg.trajectory_timeout_by_dataset.general_agent == "1200"
    assert list(cfg.algorithm.gar.sources) == ["code"] and cfg.actor_rollout_ref.rollout.n == 16
    from verl.trainer.ppo.v1.sample_mixer import MixerConfig

    mixer = MixerConfig.from_raw(cfg.trainer.v1.sampler.mixer)
    assert mixer is not None and mixer.target_basis == "accepted"
    assert mixer.sources["general"]["data_sources"] == ["mimoagent/general_agent", "mimoagent/terminal_bench"]
    advantage_fixes.check_policy_loss_config(cfg)
    advantage_fixes.check_algorithm_config(cfg)


def test_trainer_feeds_prompts_from_the_chosen_source(monkeypatch):
    """_init_mixer splits the training set by data_source; every fetched prompt comes from the
    source the mixer chose and is registered with it."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1.sample_mixer import MixerConfig
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.dataframe = {"data_source": ["opensource-code"] * 30 + ["mimoagent/general_agent"] * 10}

        def __len__(self):
            return 40

        def __getitem__(self, i):
            return {"raw_prompt": np.array([f"p{i}"], dtype=object)[0], "data_source": self.dataframe["data_source"][i], "index": i}

    class T(PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    trainer = object.__new__(T)
    trainer.config = OmegaConf.create(
        {"data": {"gen_batch_size": 1, "train_batch_size": 20, "shuffle": True, "seed": 1, "dataloader_num_workers": 0}}
    )
    trainer.parameter_sync_step = 1
    trainer.global_steps = 1
    trainer.train_dataset = DS()
    trainer.replay_buffer = type("RB", (), {})()
    trainer.mixer_config = MixerConfig(
        enable=True,
        sources={
            "code": {"data_sources": ["opensource-code"], "weight": 85},
            "general": {"data_sources": ["mimoagent/general_agent"], "weight": 15},
        },
    )
    trainer._init_mixer()
    batch = trainer._next_train_batch(20)
    sources = [trainer.mixer.groups[u].source for u in batch["uid"]]
    for uid, ds in zip(batch["uid"], batch["data_source"], strict=True):
        assert trainer.mixer.groups[uid].source == trainer.mixer.source_of_data_source(ds)
    assert set(sources) == {"code", "general"}
    forced = trainer._next_train_batch(3, source="general")
    assert {trainer.mixer.groups[u].source for u in forced["uid"]} == {"general"}

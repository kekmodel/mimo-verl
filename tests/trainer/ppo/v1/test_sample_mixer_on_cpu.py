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
"""Sample Mixer (report 6.3): Eq. 6 budgets, Eq. 7 scheduling, pause-free durations, quotas, and
the per-source replay buffer on a real TransferQueue partition."""

import heapq
import random
import uuid
from collections import Counter

import pytest

from verl.trainer.ppo.v1.sample_mixer import MixerConfig, SampleMixer, apportion, oversampling

SOURCES = {
    "code": {"data_sources": ["opensource-code"], "weight": 85, "prior_accept": 0.5, "prior_duration": 1800},
    "general": {"data_sources": ["mimoagent/general_agent", "mimoagent/terminal_bench"], "weight": 15, "prior_accept": 0.5, "prior_duration": 600},
}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _mixer(**kw):
    clock = Clock()
    return SampleMixer(MixerConfig(enable=True, sources=SOURCES, **kw), clock=clock), clock


def test_apportion_is_exact_and_proportional():
    assert apportion({"a": 85, "b": 15}, 64) == {"a": 54, "b": 10}
    assert sum(apportion({"a": 1, "b": 1, "c": 1}, 10).values()) == 10


def test_eq6_mean_oversampling_and_monotone_in_duration():
    demand = {"fast": 50.0, "slow": 10.0}
    dur = {"fast": 100.0, "slow": 2000.0}
    p = oversampling(demand, dur, p_mean=1.0, p_min=0.0, p_max=4.0)
    assert sum(demand[k] * p[k] for k in p) / sum(demand.values()) == pytest.approx(1.0, abs=1e-6)
    assert p["slow"] > p["fast"]  # slower source gets more concurrency per unit of demand
    p0 = oversampling(demand, dur, p_mean=0.0, p_min=0.0, p_max=4.0)
    assert p0 == {"fast": 0.0, "slow": 0.0}


def test_quotas_accepted_vs_generated_basis():
    m, _ = _mixer()
    m.r = {"code": 0.2, "general": 0.8}
    assert m.quotas(100) == {"code": 85, "general": 15}
    g, _ = _mixer(target_basis="generated")
    g.r = {"code": 0.2, "general": 0.8}
    # B_i ∝ π_i r_i: 85*0.2 : 15*0.8 = 17 : 12
    assert g.quotas(29) == {"code": 17, "general": 12}


def test_pause_time_is_not_active_time():
    m, clock = _mixer()
    m.on_submit("u", "code")
    clock.now = 100
    m.pause()
    clock.now = 400  # 300 s of colocated training: generation stopped
    m.resume()
    clock.now = 500
    m.r["code"], m.t["code"] = 0.0, 0.0
    m.cfg.ema = 1.0
    m.on_accepted("u")
    assert m.t["code"] == pytest.approx(200.0) and m.r["code"] == 1.0
    _, accepted = m.counts()
    assert accepted["code"] == 1
    m.on_consumed(["u"])
    assert m.counts() == ({"code": 0, "general": 0}, {"code": 0, "general": 0})


def test_eq7_round_robin_follows_the_weights_when_budgets_do_not_bind():
    m, _ = _mixer(steady_state_startup=False, p_mean=4.0, p_max=4.0)
    picks = Counter()
    for _ in range(1000):
        s = m.choose_source(100)
        picks[s] += 1
        m.on_submit(uuid.uuid4().hex, s)
        for uid in [u for u, g in m.groups.items() if g.state == "inflight"][:1]:
            m.on_rejected(uid)  # keep budgets slack
    w = m.weights(100)
    assert picks["code"] / 1000 == pytest.approx(w["code"] / sum(w.values()), abs=0.01)


def test_deficit_term_prioritizes_a_short_source():
    m, _ = _mixer(alpha=0.0, steady_state_startup=False)
    for _ in range(85):  # code already has its quota accepted and waiting
        uid = uuid.uuid4().hex
        m.on_submit(uid, "code")
        m.on_accepted(uid)
    assert m.weights(100)["code"] == 0.0
    assert all(m.choose_source(100) == "general" for _ in range(5))


def test_startup_allocates_by_duration_times_demand():
    m, _ = _mixer()
    w = m.weights(100)
    assert w["code"] / w["general"] == pytest.approx((1800 * 85) / (600 * 15))


def test_unknown_data_source_is_refused():
    m, _ = _mixer()
    with pytest.raises(ValueError, match="no mixer source"):
        m.source_of_data_source("webdev")
    with pytest.raises(ValueError, match="both"):
        MixerConfig(enable=True, sources={"a": {"data_sources": ["x"], "weight": 1}, "b": {"data_sources": ["x"], "weight": 1}})


def test_trace_simulation_fills_every_batch_without_dropping():
    """Report Fig. 16 style: sources with 10x different durations and different acceptance,
    a fixed concurrency limit, batches assembled by quota with surplus carried. Every batch must
    be filled exactly to quota, nothing accepted is ever dropped, and surplus stays bounded."""
    rng = random.Random(0)
    spec = {"code": (1800.0, 0.35), "general": (180.0, 0.8)}
    m, clock = _mixer(steady_state_startup=True)
    batch, limit = 64, 256
    events = []  # (finish_time, uid)
    accepted_uids, consumed = set(), []

    def submit():
        s = m.choose_source(batch)
        uid = uuid.uuid4().hex
        m.on_submit(uid, s)
        dur, _ = spec[s]
        heapq.heappush(events, (clock.now + rng.expovariate(1.0 / dur), uid))

    for _ in range(limit):
        submit()
    max_waiting = 0
    for _step in range(40):
        q = m.quotas(batch)
        while True:
            _, waiting = m.counts()
            if all(waiting[n] >= q[n] for n in q):
                break
            t, uid = heapq.heappop(events)
            clock.now = t
            g = m.groups[uid]
            if rng.random() < spec[g.source][1]:
                m.on_accepted(uid)
                accepted_uids.add(uid)
            else:
                m.on_rejected(uid)
            submit()  # keep the concurrency limit busy
        take = []
        for n in q:
            ready = sorted((g.submitted, u) for u, g in m.groups.items() if g.source == n and g.state == "accepted")
            take += [u for _, u in ready[: q[n]]]
        assert Counter(m.groups[u].source for u in take) == Counter(q)
        m.on_consumed(take)
        consumed += take
        max_waiting = max(max_waiting, sum(m.counts()[1].values()))
    leftover = {u for u, g in m.groups.items() if g.state == "accepted"}
    assert accepted_uids == set(consumed) | leftover  # every accepted group is trained or still waiting
    assert max_waiting < 3 * batch  # surplus does not accumulate
    assert m.r["code"] == pytest.approx(0.35, abs=0.12) and m.r["general"] == pytest.approx(0.8, abs=0.12)


# --- MixerReplayBuffer on a real TransferQueue ---------------------------------------------

tq = pytest.importorskip("transfer_queue")


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


def _rb(partition, mixer, refills):
    from verl.trainer.ppo.v1.replay_buffer import MixerReplayBuffer

    rb = MixerReplayBuffer(
        trainer_mode="colocate_async",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="wait",
        sampler_kwargs={},
        poll_interval=0.02,
        refill_fn=lambda k: k,
        filter_groups_metric="reward",
    )
    rb.mixer, rb.mixer_partition = mixer, partition

    def refill(source, k):  # the trainer registers each dispatched prompt with the mixer
        refills.append((source, k))
        for _ in range(k):
            mixer.on_submit(uuid.uuid4().hex, source)

    rb.refill_source_fn = refill
    return rb


def _put_group(partition, uid, rewards, status="finished", step=0, data_source=None):
    import torch

    for i, r in enumerate(rewards):
        tq.kv_put(
            key=f"{uid}_{i}_0",
            partition_id=partition,
            fields={"input_ids": torch.tensor([1, 2]), "rm_scores": torch.tensor([0.0, r])},
            tag={"is_prompt": False, "seq_len": 2, "global_steps": step},
        )
    fields = {"data_source": data_source} if data_source else None
    tq.kv_put(key=uid, partition_id=partition, fields=fields, tag={"is_prompt": True, "status": status, "global_steps": step})


def test_replay_buffer_takes_quotas_oldest_first_and_carries_surplus(tq_init):
    partition = f"mix-{uuid.uuid4().hex}"
    m, _ = _mixer()
    refills = []
    rb = _rb(partition, m, refills)
    groups = {}
    for i in range(6):  # 6 code groups (steps 0..5), 2 general groups
        uid = uuid.uuid4().hex
        m.on_submit(uid, "code")
        _put_group(partition, uid, [1.0, 0.0], step=i)
        groups[uid] = ("code", i)
    for i in range(2):
        uid = uuid.uuid4().hex
        m.on_submit(uid, "general")
        _put_group(partition, uid, [1.0, 0.0], step=i)
        groups[uid] = ("general", i)
    # batch 6 -> quotas 85:15 = 5 code + 1 general
    batch, _ = rb.sample(global_steps=6, partition_id=partition, batch_size=6)
    uids = {k.split("_")[0] for k in batch.keys}
    picked = Counter(groups[u][0] for u in uids)
    assert picked == Counter({"code": 5, "general": 1})
    assert max(groups[u][1] for u in uids if groups[u][0] == "code") == 4  # the 5 oldest code groups
    _, waiting = m.counts()
    assert waiting == {"code": 1, "general": 1}  # surplus carried, not dropped


def test_short_source_triggers_a_directed_dispatch_and_rejections_are_recorded(tq_init):
    import threading

    partition = f"mix-{uuid.uuid4().hex}"
    m, _ = _mixer()
    m.r["general"] = 0.5
    refills = []
    rb = _rb(partition, m, refills)
    for i in range(6):
        uid = uuid.uuid4().hex
        m.on_submit(uid, "code")
        _put_group(partition, uid, [1.0, 0.0], step=i)
    bad = uuid.uuid4().hex  # a general group with no reward variance: DAPO rejects it
    m.on_submit(bad, "general")
    _put_group(partition, bad, [1.0, 1.0])
    result = {}

    def consume():
        result["batch"], _ = rb.sample(global_steps=6, partition_id=partition, batch_size=6)

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(0.5)
    assert t.is_alive()  # waiting for general's quota
    assert bad not in m.groups  # rejected (DAPO) and recorded
    # deficit 1 at the updated acceptance estimate -> ceil(1 / r) general prompts, dispatched once
    import math

    assert refills == [("general", math.ceil(1 / m.r["general"]))]
    late = uuid.uuid4().hex
    m.on_submit(late, "general")
    _put_group(partition, late, [0.0, 1.0])
    t.join(5)
    assert not t.is_alive() and late in {k.split("_")[0] for k in result["batch"].keys}


def test_restored_groups_get_their_source_from_the_prompt_data(tq_init):
    partition = f"mix-{uuid.uuid4().hex}"
    m, _ = _mixer()
    rb = _rb(partition, m, [])
    uids = []
    for ds in ["opensource-code"] * 5 + ["mimoagent/terminal_bench"]:
        uid = uuid.uuid4().hex
        uids.append(uid)
        _put_group(partition, uid, [1.0, 0.0], data_source=ds)  # dispatched before a restart
    batch, _ = rb.sample(global_steps=1, partition_id=partition, batch_size=6)
    assert {k.split("_")[0] for k in batch.keys} == set(uids)

# GAGAR paper (arXiv 2609.32577) A.1 factor map, Flash config, checked against the Pro
# dashboard critic/code/*/advantages/max values logged in mimo-v2.6-rl-algorithm.md 13.3.
# Enumerates every pass count and tier / tied-group composition for a group of 16 and
# prints which composition reproduces each observed maximum advantage.
G = 16
F_RUN, F_MIN, F_MAX, F_LOW, LMAX = 0.9, 0.4, 0.85, 0.2, 1.5


def t2_factors(groups):  # sizes of tied groups in T2, in rank order
    K2 = len(groups)
    out = []
    for k, sz in enumerate(groups):
        f = F_MAX if K2 == 1 else F_MAX - (F_MAX - F_MIN) * k / (K2 - 1)
        out += [f] * sz
    return out


def compositions(n):
    if n == 0:
        yield []
        return
    for first in range(1, n + 1):
        for rest in compositions(n - first):
            yield [first] + rest


def adv_max(fs):
    m = len(fs)
    A = 1 - m / G
    lam = min(LMAX, m * A / sum(f * A for f in fs))
    B = [lam * f * A for f in fs] + [-m / G] * (G - m)
    return max(B) - sum(B) / G


vals = {}
for m in range(1, 16):
    for t1 in range(1, m + 1):  # the top candidate is T1
        for t1_top in range(1, t1 + 1):  # size of the tied top group in T1
            t1f = [1.0] * t1_top + [F_RUN] * (t1 - t1_top)
            for t3 in range(0, m - t1 + 1):
                t2 = m - t1 - t3
                for comp in compositions(t2) if t2 else [[]]:
                    fs = t1f + t2_factors(comp) + [F_LOW] * t3
                    vals.setdefault(round(adv_max(fs), 5), (m, t1_top, t1 - t1_top, comp, t3))

observed = {1.08333: 73, 1.04348: 10, 1.14023: 6, 1.10795: 4, 1.32344: 5, 1.26445: 3, 1.12246: 26, 1.12617: 19}
print("value     count  (passes, T1 tied-top, T1 runners, T2 tied-group sizes, T3)")
for v, n in observed.items():
    hit = [vals[u] for u in vals if abs(u - v) < 2e-5]
    print(f"{v:.5f}  x{n:3d}  {hit[0] if hit else 'not explained'}")

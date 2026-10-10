#!/usr/bin/env python3
"""
PHYSARUM ORACLE
===============
A hypothesis generator that thinks like a slime mold.

Idea (see README.md for the long version)
-----------------------------------------
Treat a knowledge graph (nodes = concepts, edges = established links) as a
network of tubes, like the body of the slime mold *Physarum polycephalum*.

  1. STRESS-TEST   Random "demand" (a source concept needs to reach a sink
                   concept) pushes flow through the tubes. Tubes that carry
                   flow thicken, idle tubes wither (Tero et al. 2010 model).
                   The graph grows a transport backbone: its own idea of
                   which connections are load-bearing.
  2. PROPOSE       Score every missing link by effective resistance in the
                   ADAPTED network (low resistance = the organism already
                   "routes" between these concepts, so a direct link is
                   plausible).
  3. SELECT        Seed the top-K proposals as thin tubes, then let the
                   organism run again under the same demand. Hypotheses
                   compete for flow. The ones that get reinforced are
                   "vigorous"; the ones that starve are withered.

Output = a ranked list of hypotheses, each with a plausibility score and a
survival vigor, plus pictures of the organism thinking.

Usage
-----
    python physarum_oracle.py demo                  # GIF + hero figure on a toy world
    python physarum_oracle.py benchmark --seeds 20  # honest comparison vs. baselines
    python physarum_oracle.py tune                  # hyper-parameter search (validation seeds)
    python physarum_oracle.py discover --edges my_edges.csv   # run on your own graph
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, replace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# --------------------------------------------------------------------------- #
# Look & feel (dark surface; series colours from a validated categorical palette)
# --------------------------------------------------------------------------- #
BG, SURFACE = "#0d0d0d", "#1a1a19"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
HYP = "#e87ba4"
METHOD_COLORS = {
    "Physarum resistance": "#3987e5",
    "Raw resistance": "#d95926",
    "Common neighbours": "#199e70",
    "Adamic-Adar": "#c98500",
    "Resource allocation": "#d55181",
}
FLOW_CMAP = LinearSegmentedColormap.from_list(
    "flow", ["#2a3140", "#184f95", "#3987e5", "#9ec5f4", "#f0efec"]
)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    # organism dynamics
    steps: int = 160          # stress-test iterations
    batch: int = 24           # source/sink pairs per iteration
    i0: float = 1.0           # current injected per pair
    gamma: float = 1.5        # flow nonlinearity (>1 = winner-take-all)
    mu: float = 0.3           # decay rate of idle tubes
    dt: float = 0.25
    d0: float = 0.5           # initial tube conductance
    d_floor: float = 1e-3
    conserve: bool = True     # fixed biomass: rescale so mean tube size stays constant
    p_local: float = 0.9      # fraction of demand that is "local" (within local_hops)
    local_hops: int = 3
    # scoring
    prior: float = 0.6        # blend of uniform conductance into the adapted network
    # selection
    top_k: int = 40
    seed_frac: float = 0.3    # hypothesis tubes start at this fraction of mean tube size
    sel_steps: int = 120


# --------------------------------------------------------------------------- #
# Graph utilities
# --------------------------------------------------------------------------- #
def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    return np.array([find(i) for i in range(n)])


def is_connected(n, edges):
    return len(set(components(n, edges).tolist())) == 1


def make_world(n=150, k=6, seed=0, sigma_c=0.45, sigma_l=0.6, p0=0.85):
    """Toy 'scientific landscape': k fields (clusters) on a ring; links are
    dense inside a field and rare across fields. Returns edges, 2D positions,
    and field labels."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, k)
    centres = 2.2 * np.c_[np.cos(ang), np.sin(ang)]
    comm = rng.integers(0, k, n)
    pos = centres[comm] + rng.normal(0, sigma_c, (n, 2))
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    p = np.triu(p0 * np.exp(-d2 / (2 * sigma_l**2)), 1)
    A = rng.random((n, n)) < p
    edges = np.argwhere(np.triu(A, 1))
    # stitch components together through the geometrically closest pair
    while True:
        comp = components(n, edges)
        if len(set(comp.tolist())) == 1:
            break
        dd = np.sqrt(d2)
        dd[comp[:, None] == comp[None, :]] = np.inf
        i, j = np.unravel_index(np.argmin(dd), dd.shape)
        edges = np.vstack([edges, [min(i, j), max(i, j)]])
    return edges.astype(int), pos, comm


def holdout(n, edges, frac, rng):
    """Remove `frac` of the edges without disconnecting the graph."""
    m = len(edges)
    k = int(round(frac * m))
    keep = np.ones(m, bool)
    removed = []
    for idx in rng.permutation(m):
        if len(removed) >= k:
            break
        keep[idx] = False
        if is_connected(n, edges[keep]):
            removed.append(idx)
        else:
            keep[idx] = True
    return edges[keep], edges[removed]


def adjacency(n, edges):
    A = np.zeros((n, n))
    A[edges[:, 0], edges[:, 1]] = 1
    A[edges[:, 1], edges[:, 0]] = 1
    return A


def incidence(edges, n):
    inc = np.zeros((len(edges), n))
    r = np.arange(len(edges))
    inc[r, edges[:, 0]] = 1
    inc[r, edges[:, 1]] = -1
    return inc


def hop_matrix(n, edges, cap=99):
    A = adjacency(n, edges).astype(np.int32)
    H = np.full((n, n), cap, int)
    np.fill_diagonal(H, 0)
    reach = np.eye(n, dtype=bool)
    frontier = reach.copy()
    h = 0
    while frontier.any():
        h += 1
        nxt = ((frontier.astype(np.int32) @ A) > 0) & ~reach
        H[nxt] = h
        reach |= nxt
        frontier = nxt
    return H


def spring_layout(n, edges, seed=0, iters=300):
    """Tiny Fruchterman-Reingold layout (used when no coordinates are known)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1, (n, 2))
    k = 1.6 / np.sqrt(n)
    temp = 0.2
    u, v = edges[:, 0], edges[:, 1]
    for _ in range(iters):
        delta = pos[:, None] - pos[None]
        dist = np.linalg.norm(delta, axis=-1) + 1e-3
        disp = ((k * k / dist**2)[..., None] * delta).sum(1)
        d = pos[u] - pos[v]
        dd = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        att = (dd / k) * d
        np.add.at(disp, u, -att)
        np.add.at(disp, v, att)
        length = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        pos += disp / length * np.minimum(length, temp)
        temp *= 0.985
    return pos


def normalise_positions(pos):
    pos = pos - pos.mean(0)
    return pos / np.abs(pos).max()


# --------------------------------------------------------------------------- #
# The organism
# --------------------------------------------------------------------------- #
def sample_pairs(rng, n, p, local):
    s = rng.integers(0, n, p.batch)
    t = np.empty(p.batch, int)
    for b in range(p.batch):
        if rng.random() < p.p_local and len(local[s[b]]):
            t[b] = rng.choice(local[s[b]])
        else:
            t[b] = rng.integers(0, n)
            while t[b] == s[b]:
                t[b] = rng.integers(0, n)
    return s, t


def adapt(edges, n, D0, local, p, rng, steps, record=0, tail=20):
    """Physarum adaptation: dD/dt = f(|Q|) - mu*D, driven by random source/sink demand.
    Returns final D, recorded frames [(D, meanflow)], and D averaged over the last `tail` steps."""
    inc = incidence(edges, n)
    D = D0.copy()
    mass = D0.sum()
    cols = np.arange(p.batch)
    frames, acc, cnt = [], np.zeros_like(D), 0
    for it in range(steps):
        s, t = sample_pairs(rng, n, p, local)
        L = inc.T @ (inc * D[:, None])
        rhs = np.zeros((n, p.batch))
        rhs[s, cols] += p.i0
        rhs[t, cols] -= p.i0
        P = np.zeros((n, p.batch))
        P[1:] = np.linalg.solve(L[1:, 1:], rhs[1:])  # node 0 is ground
        q = np.abs(D[:, None] * (inc @ P)).mean(1)
        qn = q / q.mean()  # scale-free: flow relative to the average tube
        D = np.maximum(D + p.dt * (qn**p.gamma / (1 + qn**p.gamma) - p.mu * D), p.d_floor)
        if p.conserve:
            D = np.maximum(D * (mass / D.sum()), p.d_floor)
        if record and (it % record == 0 or it == steps - 1):
            frames.append((D.copy(), q.copy()))
        if it >= steps - tail:
            acc += D
            cnt += 1
    return D, frames, acc / max(cnt, 1)


def resistance(edges, n, w):
    inc = incidence(edges, n)
    L = inc.T @ (inc * w[:, None])
    Lp = np.linalg.pinv(L, hermitian=True)
    d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2 * Lp


# --------------------------------------------------------------------------- #
# Scoring, baselines, metrics
# --------------------------------------------------------------------------- #
def score_all(n, edges, D, p):
    A = adjacency(n, edges)
    deg = A.sum(1)
    scores = {
        "Common neighbours": A @ A,
        "Adamic-Adar": (A / np.log(np.maximum(deg, 2))[None, :]) @ A,
        "Resource allocation": (A / np.maximum(deg, 1)[None, :]) @ A,
        "Raw resistance": -resistance(edges, n, np.ones(len(edges))),
    }
    w = (1 - p.prior) * D / D.mean() + p.prior
    scores["Physarum resistance"] = -resistance(edges, n, w)
    return scores


def auc_score(pos, neg):
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def ranked(flat_scores, mask, rng):
    s = np.where(mask, flat_scores, -np.inf)
    return np.lexsort((rng.random(len(s)), -s))  # best first, random tie-break


def local_lists(hops, kmax):
    return [np.flatnonzero((hops[i] > 0) & (hops[i] <= kmax)) for i in range(len(hops))]


# --------------------------------------------------------------------------- #
# The full pipeline
# --------------------------------------------------------------------------- #
def run_oracle(n, edges, p, seed=0, record=False, with_selection=True):
    rng = np.random.default_rng(seed)
    hops = hop_matrix(n, edges)
    local = local_lists(hops, p.local_hops)
    D, framesA, _ = adapt(edges, n, np.full(len(edges), p.d0), local, p, rng, p.steps,
                          record=4 if record else 0)
    S = score_all(n, edges, D, p)
    iu = np.triu_indices(n, 1)
    A = adjacency(n, edges)
    cand_mask = A[iu] == 0
    flat = {k: v[iu] for k, v in S.items()}
    out = dict(D=D, framesA=framesA, S=S, flat=flat, iu=iu, cand_mask=cand_mask, hops=hops)
    if not with_selection:
        return out

    order = ranked(flat["Physarum resistance"], cand_mask, rng)[: p.top_k]
    cand = np.c_[iu[0][order], iu[1][order]]
    ext = np.vstack([edges, cand])
    m0 = len(edges)
    d_seed = p.seed_frac * D.mean()
    D_ext, framesB, D_avg = adapt(ext, n, np.concatenate([D, np.full(len(cand), d_seed)]),
                                  local, p, rng, p.sel_steps, record=4 if record else 0)
    base = D_avg[:m0].mean()
    vigor = D_avg[m0:] / base
    plaus = flat["Physarum resistance"][order]
    out.update(cand=cand, cand_flat_idx=order, plaus=plaus, vigor=vigor, d_seed=d_seed, base=base,
               framesB=framesB, D_ext=D_ext, m0=m0,
               order_plaus=np.argsort(-plaus, kind="stable"), order_vigor=np.argsort(-vigor, kind="stable"))
    return out


def evaluate_world(seed, p, n=150, frac=0.15):
    edges, pos, comm = make_world(n=n, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    train, held = holdout(n, edges, frac, rng)
    res = run_oracle(n, train, p, seed=seed)
    iu = res["iu"]
    T = adjacency(n, edges)[iu] > 0
    cm = res["cand_mask"]
    pos_m, neg_m = cm & T, cm & ~T
    row = {"seed": seed, "n_edges": int(len(edges)), "n_heldout": int(len(held)), "auc": {}, "pk": {}}
    ks = list(range(5, 55, 5))
    for name, f in res["flat"].items():
        row["auc"][name] = auc_score(f[pos_m], f[neg_m])
        order = ranked(f, cm, np.random.default_rng(seed + 20_000))
        row["pk"][name] = [float(T[order[:k]].mean()) for k in ks]
    cand_truth = T[res["cand_flat_idx"]]
    row["pool_precision"] = float(cand_truth.mean())
    for nm, key in (("plausibility", "order_plaus"), ("vigor", "order_vigor")):
        row[f"p10_{nm}"] = float(cand_truth[res[key][:10]].mean())
    row["vigor_truth_corr"] = float(np.corrcoef(np.log(res["vigor"] + 1e-9), cand_truth.astype(float))[0, 1]) \
        if 0 < cand_truth.sum() < len(cand_truth) else float("nan")
    row["ks"] = ks
    return row


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def arc(p0, p1, bend=0.14, N=26):
    mid = (p0 + p1) / 2
    d = p1 - p0
    perp = np.array([-d[1], d[0]])
    c = mid + bend * perp
    t = np.linspace(0, 1, N)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t**2 * p1


def draw_scene(ax, pos, edges, D, q, dref, qref, hyp=None, lw=1.0, deg=None):
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    segs = pos[edges]
    qn = np.clip(q / qref, 0, 1)
    cols = FLOW_CMAP(qn)
    cols[:, 3] = 0.15 + 0.80 * qn**0.7
    wd = lw * (0.25 + 5.0 * np.clip(D / dref, 0, 1.25) ** 0.9)
    order = np.argsort(qn)  # brightest drawn last
    for mult, al in ((3.6, 0.06), (1.9, 0.13), (1.0, 1.0)):
        c = cols[order].copy()
        c[:, 3] *= al
        ax.add_collection(LineCollection(segs[order], colors=c, linewidths=wd[order] * mult,
                                         capstyle="round", zorder=2))
    if deg is None:
        deg = np.bincount(edges.ravel(), minlength=len(pos))
    sz = 5 + 2.6 * deg
    ax.scatter(pos[:, 0], pos[:, 1], s=sz * 7 * lw, color=INK2, alpha=0.06, zorder=3, linewidths=0)
    ax.scatter(pos[:, 0], pos[:, 1], s=sz * lw, color=INK2, alpha=0.9, zorder=4, linewidths=0)
    if hyp is not None and len(hyp["pairs"]):
        pr = hyp["pairs"]
        polys = [arc(pos[a], pos[b]) for a, b in pr]
        wh = lw * (0.7 + 5.0 * np.clip(hyp["D"] / dref, 0, 1.25) ** 0.9)
        al = hyp["alpha"]
        styles = hyp.get("styles", ["-"] * len(pr))
        base = np.array(matplotlib.colors.to_rgba(HYP))
        for mult, a_mul in ((3.4, 0.10), (1.8, 0.18), (1.0, 1.0)):
            c = np.tile(base, (len(pr), 1))
            c[:, 3] = al * a_mul
            ax.add_collection(LineCollection(polys, colors=c, linewidths=wh * mult, linestyles=styles,
                                             capstyle="round", zorder=5))
        ring = hyp.get("ring")
        if ring is not None and ring.any():
            pts = np.array([pos[x] for (a, b), r in zip(pr, ring) if r for x in (a, b)])
            ax.scatter(pts[:, 0], pts[:, 1], s=60 * lw, facecolors="none", edgecolors=INK,
                       linewidths=1.0 * lw, zorder=6)


def hud(fig, title, sub, lines=()):
    fig.text(0.04, 0.955, title, color=INK, fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.04, 0.915, sub, color=INK2, fontsize=9.2, va="top")
    for i, s in enumerate(lines):
        fig.text(0.04, 0.045 + 0.034 * (len(lines) - 1 - i), s, color=MUTED, fontsize=8.6, va="bottom")


def frame_image(pos, edges, D, q, refs, title, sub, lines=(), hyp=None, size=6.0, dpi=90):
    fig = plt.figure(figsize=(size, size), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    draw_scene(ax, pos, edges, D, q, refs[0], refs[1], hyp=hyp)
    hud(fig, title, sub, lines)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return img


def make_gif(path, pos, train, res, p, truth_pairs=None):
    D_final, q_final = res["framesA"][-1]
    refs = (np.percentile(D_final, 98), max(np.percentile(q_final, 97), 1e-6))
    m0, K = res["m0"], len(res["cand"])
    frames, durations = [], []
    alive_thr = 0.15 * refs[0]

    def add(img, ms):
        frames.append(img)
        durations.append(ms)

    for i, (D, q) in enumerate(res["framesA"]):
        it = min(i * 4, p.steps)
        add(frame_image(pos, train, D, q, refs, "PHYSARUM ORACLE",
                        "1 / 3   Stress-testing the graph: flow carves a backbone",
                        [f"iteration {it:3d}   tubes alive {int((D > alive_thr).sum())}/{len(D)}"]), 70)
    D, q = res["framesA"][-1]
    d_seed = res["d_seed"]
    for f in np.linspace(0.12, 1, 8):
        hyp = dict(pairs=res["cand"], D=np.full(K, d_seed), alpha=np.full(K, 0.9 * f))
        add(frame_image(pos, train, D, q, refs, "PHYSARUM ORACLE",
                        "2 / 3   Seeding the top proposals as thin tubes",
                        [f"{K} hypotheses seeded"], hyp=hyp), 90)
    for i, (De, qe) in enumerate(res["framesB"]):
        it = min(i * 4, p.sel_steps)
        hyp = dict(pairs=res["cand"], D=De[m0:], alpha=np.clip(0.25 + 0.75 * De[m0:] / refs[0] * 1.6, 0.2, 1))
        add(frame_image(pos, train, De[:m0], qe[:m0], refs, "PHYSARUM ORACLE",
                        "3 / 3   Selection: hypotheses compete for flow",
                        [f"iteration {it:3d}   survivors {int((De[m0:] >= d_seed).sum())}/{K}"], hyp=hyp), 70)
    De, qe = res["framesB"][-1]
    if truth_pairs is not None:
        tp = {tuple(x) for x in truth_pairs.tolist()}
        conf = np.array([(int(a), int(b)) in tp for a, b in res["cand"]])
        n_conf = int(conf.sum())
        for f in np.linspace(0, 1, 6):
            alpha = np.where(conf, 1.0, 1 - 0.7 * f)
            hyp = dict(pairs=res["cand"], D=De[m0:], alpha=alpha, ring=conf if f > 0.5 else None,
                       styles=["-" if c else "--" for c in conf])
            add(frame_image(pos, train, De[:m0], qe[:m0], refs, "PHYSARUM ORACLE",
                            "Reveal: held-out ground truth (circled = real link)",
                            [f"{n_conf} of {K} proposals are real links the model never saw "
                             f"({100 * n_conf / K:.0f}%)"], hyp=hyp), 160)
        durations[-1] = 2500
    else:
        durations[-1] = 1800
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def short(names, i):
    return names[i]


def hero_figure(path, pos, train, res, p, names, truth_pairs=None):
    m0, K = res["m0"], len(res["cand"])
    D, q = res["framesA"][-1] if res["framesA"] else (res["D"], np.ones_like(res["D"]))
    De = res["D_ext"]
    refs = (np.percentile(res["D"], 98), max(np.percentile(q, 97), 1e-6))
    conf = None
    if truth_pairs is not None:
        tp = {tuple(x) for x in truth_pairs.tolist()}
        conf = np.array([(int(a), int(b)) in tp for a, b in res["cand"]])
    vig = res["vigor"]
    alpha = np.clip(0.25 + 0.75 * De[m0:] / refs[0] * 1.6, 0.2, 1)
    styles = ["-" if (conf is None or c) else "--" for c in (conf if conf is not None else np.ones(K, bool))]
    hyp = dict(pairs=res["cand"], D=De[m0:], alpha=alpha, styles=styles, ring=conf)

    fig = plt.figure(figsize=(16, 9), dpi=130, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 0.60, 1.0], facecolor=BG)
    draw_scene(ax, pos, train, res["D"], q, refs[0], refs[1], hyp=hyp, lw=1.5)
    fig.text(0.03, 0.955, "PHYSARUM ORACLE", color=INK, fontsize=20, fontweight="bold", va="top")
    fig.text(0.03, 0.915, "a slime mold that reads a knowledge graph and grows the links it thinks are missing",
             color=INK2, fontsize=11, va="top")
    lg = [("tube width = learned conductance (how load-bearing a link is)", None),
          ("pink arcs = hypotheses; width = how much flow they won", None)]
    if conf is not None:
        lg.append(("solid + circled = real held-out link   dashed = not in ground truth", None))
    for i, (s, _) in enumerate(lg):
        fig.text(0.03, 0.05 + 0.028 * (len(lg) - 1 - i), s, color=MUTED, fontsize=9.5, va="bottom")

    # scatter: plausibility vs vigor
    sc = fig.add_axes([0.665, 0.56, 0.30, 0.32], facecolor=SURFACE)
    allp = res["flat"]["Physarum resistance"][res["cand_mask"]]
    top_pct = np.array([100 * (allp > v).mean() for v in res["plaus"]])  # "in the top x% of missing links"
    reff = -res["plaus"]
    for s_ in sc.spines.values():
        s_.set_color(AXIS)
        s_.set_linewidth(0.8)
    sc.set_yscale("log")
    sc.invert_xaxis()  # right = more plausible
    sc.grid(color=GRID, linewidth=0.6)
    sc.set_axisbelow(True)
    sc.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 4)))
    if conf is None:
        sc.scatter(reff, vig, s=46, color=HYP, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    else:
        sc.scatter(reff[conf], vig[conf], s=52, color=HYP, edgecolors=SURFACE, linewidths=1.2, zorder=3,
                   label="real (held-out)")
        sc.scatter(reff[~conf], vig[~conf], s=52, facecolors="none", edgecolors=HYP, linewidths=1.4,
                   zorder=3, label="not in ground truth")
        sc.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  labelcolor=INK2, fontsize=8.5, handletextpad=0.3, columnspacing=1.4)
    safe = [int(i) for i in res["order_plaus"][:6]]
    bold = [int(i) for i in res["order_vigor"] if int(i) not in safe][:6]
    for tag, group in (("S", safe), ("B", bold)):
        for r, i in enumerate(group, 1):
            sc.annotate(f"{tag}{r}", (reff[i], vig[i]), xytext=(5, 4), textcoords="offset points",
                        color=INK, fontsize=8.5)
    sc.set_xlabel("effective resistance in the adapted network  (lower = more plausible →)",
                  color=INK2, fontsize=9)
    sc.set_ylabel("vigor: flow won (log; 1 = avg tube)", color=INK2, fontsize=9)
    sc.tick_params(colors=MUTED, labelsize=8.5)
    sc.set_title("Hypothesis map: safe bets vs bold bridges", color=INK, fontsize=11, loc="left", pad=26)

    # table
    tb = fig.add_axes([0.655, 0.05, 0.33, 0.43], facecolor=BG)
    tb.axis("off")
    tb.set_xlim(0, 1)
    tb.set_ylim(0, 1)
    hdr = "  id   pair         top%    vigor" + ("   real" if conf is not None else "")
    rowh, y = 0.058, 0.97
    for title, tag, group in (("SAFE BETS  (most plausible)", "S", safe),
                              ("BOLD BRIDGES  (won the most flow)", "B", bold)):
        tb.text(0, y, title, color=INK, fontsize=10.5, va="top")
        y -= rowh
        tb.text(0, y, hdr, color=MUTED, fontsize=8.8, va="top", family="DejaVu Sans Mono")
        y -= rowh
        for r, i in enumerate(group, 1):
            a, b = res["cand"][i]
            mark = ("   ✓" if conf[i] else "   ·") if conf is not None else ""
            tb.text(0, y, f"  {tag}{r}   {names[a]:>4s}–{names[b]:<4s}   {top_pct[i]:5.2f}   {vig[i]:5.2f}{mark}",
                    color=INK2, fontsize=9.3, va="top", family="DejaVu Sans Mono")
            y -= rowh * 0.92
        y -= rowh * 0.5
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --- benchmark figure ------------------------------------------------------- #
def style_axes(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for k, s_ in ax.spines.items():
        s_.set_visible(k == "bottom" or k == "left")
        s_.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def benchmark_figure(path, rows):
    methods = list(METHOD_COLORS)
    aucs = {m: np.array([r["auc"][m] for r in rows]) for m in methods}
    ks = rows[0]["ks"]
    pk = {m: np.array([r["pk"][m] for r in rows]) for m in methods}
    fig = plt.figure(figsize=(15.5, 5.6), dpi=130, facecolor=BG)
    # A
    ax = fig.add_axes([0.14, 0.2, 0.22, 0.62])
    style_axes(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    order = sorted(methods, key=lambda m: aucs[m].mean())
    for i, m in enumerate(order):
        mu, sd = aucs[m].mean(), aucs[m].std()
        ax.barh(i, mu - 0.5, left=0.5, height=0.5, color=METHOD_COLORS[m], zorder=3)
        ax.plot([mu - sd, mu + sd], [i, i], color=INK, lw=1.0, zorder=4)
        ax.text(mu + sd + 0.006, i, f"{mu:.3f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color=INK2, fontsize=9.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC (0.5 = chance)", color=INK2, fontsize=9.5)
    ax.set_title("Ranking held-out links: AUC", color=INK, fontsize=11.5, loc="left", pad=10)
    # B
    ax = fig.add_axes([0.46, 0.2, 0.23, 0.62])
    style_axes(ax)
    for m in methods:
        mu = pk[m].mean(0)
        ax.plot(ks, mu, color=METHOD_COLORS[m], lw=2.0, marker="o", ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=m, zorder=3)
    ax.set_xlabel("K (top-K proposals)", color=INK2, fontsize=9.5)
    ax.set_ylabel("precision@K", color=INK2, fontsize=9.5)
    ax.set_ylim(0, None)
    ax.set_title("Precision of the top-K proposals", color=INK, fontsize=11.5, loc="left", pad=10)
    # C
    ax = fig.add_axes([0.77, 0.2, 0.2, 0.62])
    style_axes(ax)
    labs = ["pool average\n(40 proposals)", "top 10 by\nplausibility", "top 10 by\nvigor"]
    keys = ["pool_precision", "p10_plausibility", "p10_vigor"]
    cols = [MUTED, METHOD_COLORS["Physarum resistance"], METHOD_COLORS["Resource allocation"]]
    vals = [np.mean([r[k] for r in rows]) for k in keys]
    sds = [np.std([r[k] for r in rows]) for k in keys]
    for i, (v, s_) in enumerate(zip(vals, sds)):
        ax.bar(i, v, width=0.5, color=cols[i], zorder=3)
        ax.plot([i, i], [max(v - s_, 0), v + s_], color=INK, lw=1.0, zorder=4)
        ax.text(i + 0.3, v, f"{v:.2f}", ha="left", va="center", color=INK, fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, color=INK2, fontsize=9.5)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylim(0, None)
    ax.set_ylabel("fraction that are real held-out links", color=INK2, fontsize=9.5)
    ax.set_title("Plausibility vs vigor: a trade-off", color=INK, fontsize=11.5, loc="left", pad=10)
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, labelcolor=INK2, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.03, 0.95, f"Held-out link prediction on {len(rows)} random toy worlds "
             f"(15% of links hidden; mean ± 1 sd)", color=INK, fontsize=13, fontweight="bold", va="top")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def load_edges(path):
    names, idx, pairs = [], {}, []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if not a or not b or a == b or (a.lower() in ("source", "src", "from") and not pairs):
                continue
            for x in (a, b):
                if x not in idx:
                    idx[x] = len(names)
                    names.append(x)
            pairs.append((min(idx[a], idx[b]), max(idx[a], idx[b])))
    edges = np.array(sorted(set(pairs)), dtype=int)
    n = len(names)
    comp = components(n, edges)
    giant = np.bincount(comp).argmax() if comp.max() < n else max(set(comp.tolist()), key=lambda c: (comp == c).sum())
    keep = np.flatnonzero(comp == giant)
    if len(keep) < n:
        print(f"note: using the largest connected component ({len(keep)} of {n} nodes)")
        remap = -np.ones(n, int)
        remap[keep] = np.arange(len(keep))
        edges = edges[(comp[edges[:, 0]] == giant) & (comp[edges[:, 1]] == giant)]
        edges = remap[edges]
        names = [names[i] for i in keep]
    return names, edges


def write_hypotheses(path, res, names, conf=None):
    allp = res["flat"]["Physarum resistance"][res["cand_mask"]]
    vig_rank = {int(i): r for r, i in enumerate(res["order_vigor"], 1)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plausibility_rank", "concept_a", "concept_b", "top_percent_of_missing_links",
                    "vigor", "vigor_rank", "survives", "bucket"]
                   + (["in_ground_truth"] if conf is not None else []))
        for r, i in enumerate(res["order_plaus"], 1):
            i = int(i)
            a, b = res["cand"][i]
            top = 100 * (allp > res["plaus"][i]).mean()
            bucket = "safe" if r <= 10 else ("bold" if vig_rank[i] <= 10 else "")
            row = [r, names[a], names[b], f"{top:.3f}", f"{res['vigor'][i]:.3f}", vig_rank[i],
                   int(res["vigor"][i] * res["base"] >= res["d_seed"]), bucket]
            if conf is not None:
                row.append(int(conf[i]))
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_demo(a):
    os.makedirs(a.out, exist_ok=True)
    p = Params()
    t0 = time.time()
    n = 150
    edges, pos, comm = make_world(n=n, seed=a.seed)
    rng = np.random.default_rng(a.seed + 10_000)
    train, held = holdout(n, edges, 0.15, rng)
    print(f"world: {n} concepts, {len(edges)} links; hid {len(held)} for the reveal")
    res = run_oracle(n, train, p, seed=a.seed, record=True)
    pos = normalise_positions(pos)
    names = [f"{'ABCDEF'[comm[i]]}{i:02d}" for i in range(n)]
    tset = {tuple(x) for x in held.tolist()}
    conf = np.array([(int(x), int(y)) in tset for x, y in res["cand"]])
    print(f"top-{p.top_k} pool: {conf.sum()} real links ({100 * conf.mean():.0f}%) "
          f"vs base rate {100 * len(held) / (n * (n - 1) / 2 - len(train)):.2f}%")
    hero_figure(os.path.join(a.out, "oracle_map.png"), pos, train, res, p, names, truth_pairs=held)
    print("hero figure done", f"{time.time() - t0:.1f}s")
    make_gif(os.path.join(a.out, "organism.gif"), pos, train, res, p, truth_pairs=held)
    write_hypotheses(os.path.join(a.out, "hypotheses.csv"), res, names, conf=conf)
    with open(os.path.join(a.out, "toy_world_edges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        for x, y in train:
            w.writerow([names[x], names[y]])
    print(f"done in {time.time() - t0:.1f}s -> {a.out}")


def cmd_benchmark(a):
    os.makedirs(a.out, exist_ok=True)
    p = Params()
    rows = []
    t0 = time.time()
    for s in range(a.seeds):
        rows.append(evaluate_world(s, p))
        print(f"seed {s:2d}  AUC phys {rows[-1]['auc']['Physarum resistance']:.3f}  "
              f"CN {rows[-1]['auc']['Common neighbours']:.3f}  ({time.time() - t0:.0f}s)")
    benchmark_figure(os.path.join(a.out, "benchmark.png"), rows)
    with open(os.path.join(a.out, "benchmark_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n=== summary (mean ± sd over seeds) ===")
    for m in METHOD_COLORS:
        v = np.array([r["auc"][m] for r in rows])
        p10 = np.array([r["pk"][m][1] for r in rows])
        print(f"{m:22s} AUC {v.mean():.3f} ± {v.std():.3f}   P@10 {p10.mean():.2f}")
    for k in ("pool_precision", "p10_plausibility", "p10_vigor"):
        v = np.array([r[k] for r in rows])
        print(f"{k:22s} {v.mean():.3f} ± {v.std():.3f}")
    c = np.array([r["vigor_truth_corr"] for r in rows])
    print(f"corr(log vigor, real) within pool: {np.nanmean(c):.3f} (n={np.sum(~np.isnan(c))})")
    # paired comparison phys vs best baseline
    base = max((m for m in METHOD_COLORS if m != "Physarum resistance"),
               key=lambda m: np.mean([r["auc"][m] for r in rows]))
    diff = np.array([r["auc"]["Physarum resistance"] - r["auc"][base] for r in rows])
    print(f"paired AUC diff vs best baseline ({base}): {diff.mean():+.4f}  "
          f"(wins {(diff > 0).sum()}/{len(diff)}, se {diff.std(ddof=1) / np.sqrt(len(diff)):.4f})")


def cmd_tune(a):
    base = Params()
    seeds = list(range(1000, 1000 + a.seeds))
    worlds = []
    for s in seeds:
        edges, _, _ = make_world(n=150, seed=s)
        train, held = holdout(150, edges, 0.15, np.random.default_rng(s + 10_000))
        worlds.append((edges, train))
    print("Validation seeds only (never used for the reported benchmark).")
    print("gamma  mu  p_local prior | val AUC | raw AUC | contrast D95/D25")
    results = []
    for g in (1.0, 1.5, 2.5):
        for mu in (0.3, 1.0):
            for pl in (0.5, 0.9):
                for pr in (0.3, 0.6):
                    p = replace(base, gamma=g, mu=mu, p_local=pl, prior=pr)
                    aucs, raws, con = [], [], []
                    for (edges, train), s in zip(worlds, seeds):
                        res = run_oracle(150, train, p, seed=s, with_selection=False)
                        T = adjacency(150, edges)[res["iu"]] > 0
                        cm = res["cand_mask"]
                        f = res["flat"]["Physarum resistance"]
                        aucs.append(auc_score(f[cm & T], f[cm & ~T]))
                        fr = res["flat"]["Raw resistance"]
                        raws.append(auc_score(fr[cm & T], fr[cm & ~T]))
                        con.append(np.percentile(res["D"], 95) / np.percentile(res["D"], 25))
                    results.append((np.mean(aucs), g, mu, pl, pr))
                    print(f"{g:4.1f} {mu:4.1f} {pl:5.1f} {pr:5.1f} | {np.mean(aucs):.4f} | "
                          f"{np.mean(raws):.4f} | {np.mean(con):7.1f}", flush=True)
    best = max(results)
    print(f"\nhighest validation AUC: {best[0]:.4f} at gamma={best[1]} mu={best[2]} "
          f"p_local={best[3]} prior={best[4]}")
    print("Shipped defaults (gamma=1.5, mu=0.3, p_local=0.9, prior=0.6) are chosen for a visibly "
          "adapted backbone (contrast ~4x), not for peak AUC. See README 'Honest results'.")


def cmd_discover(a):
    os.makedirs(a.out, exist_ok=True)
    names, edges = load_edges(a.edges)
    n = len(names)
    print(f"graph: {n} concepts, {len(edges)} links")
    p = replace(Params(), top_k=a.top)
    res = run_oracle(n, edges, p, seed=a.seed, record=a.gif)
    pos = normalise_positions(spring_layout(n, edges, seed=a.seed))
    if not res["framesA"]:
        D = res["D"]
        res["framesA"] = [(D, np.ones_like(D))]
    write_hypotheses(os.path.join(a.out, "hypotheses.csv"), res, names)
    hero_figure(os.path.join(a.out, "oracle_map.png"), pos, edges, res, p, names)
    if a.gif:
        make_gif(os.path.join(a.out, "organism.gif"), pos, edges, res, p)
    print(f"wrote hypotheses.csv and figures to {a.out}")
    print("\nsafe bets (most plausible):")
    for r, i in enumerate(res["order_plaus"][:8], 1):
        x, y = res["cand"][i]
        print(f"  S{r}  {names[x]} -- {names[y]}   vigor {res['vigor'][i]:.2f}")
    print("bold bridges (won the most flow):")
    for r, i in enumerate(res["order_vigor"][:8], 1):
        x, y = res["cand"][i]
        print(f"  B{r}  {names[x]} -- {names[y]}   vigor {res['vigor'][i]:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", default="outputs")
    d.set_defaults(fn=cmd_demo)
    b = sub.add_parser("benchmark")
    b.add_argument("--seeds", type=int, default=20)
    b.add_argument("--out", default="outputs")
    b.set_defaults(fn=cmd_benchmark)
    t = sub.add_parser("tune")
    t.add_argument("--seeds", type=int, default=6)
    t.set_defaults(fn=cmd_tune)
    c = sub.add_parser("discover")
    c.add_argument("--edges", required=True, help="CSV with two columns: concept_a,concept_b")
    c.add_argument("--top", type=int, default=40)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--gif", action="store_true")
    c.add_argument("--out", default="outputs/discover")
    c.set_defaults(fn=cmd_discover)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the paper's figures from the pooled results.
Outputs PDF (for LaTeX) and PNG (for visual check) into paper/figures/.

Fig 1  mechanism   : clipped-Gaussian schematic (the impossible overestimate tail).
Fig 2  collapse    : measured gain vs a=(d-1)/sigma, all m, on the closed-form curve.
Fig 3  window      : gain vs duplicate fraction for each m (the value window narrows).
Fig 4  universality: d=1.0 clip gain across 5 sketch families + real-world PyPI.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
PH1 = os.path.join(os.path.dirname(HERE), "phase1")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
})
phi, Phi, Phibar = norm.pdf, norm.cdf, norm.sf
C = {"blue": "#2b6cb0", "red": "#c53030", "green": "#2f855a",
     "orange": "#dd6b20", "purple": "#6b46c1", "gray": "#4a5568"}


def mse_ratio(a):
    return Phi(a) - a * phi(a) + a ** 2 * Phibar(a)


def gain(a):
    return 100 * (1 - np.sqrt(mse_ratio(a)))


def load(name):
    with open(os.path.join(PH1, name)) as f:
        return json.load(f)


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, stem + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", stem)


# ---------------------------------------------------------------- Fig 1
def fig_mechanism():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.linspace(0.86, 1.14, 600)
    sig = 0.033
    y = norm.pdf(x, 1.0, sig)
    ax.plot(x, y, color=C["blue"], lw=1.6)
    tail = x >= 1.0
    ax.fill_between(x[tail], 0, y[tail], color=C["red"], alpha=0.30,
                    label="overestimates $>L$\n(impossible)")
    ax.fill_between(x[~tail], 0, y[~tail], color=C["blue"], alpha=0.12)
    ax.axvline(1.0, color=C["gray"], lw=1.0, ls="--")
    ax.annotate("$L=n$", xy=(1.0, norm.pdf(1.0, 1.0, sig)), xytext=(1.05, 11),
                fontsize=9, color=C["gray"])
    ax.set_xlabel(r"estimate $\hat{n}\,/\,n$")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.set_title("Half the error mass is provably impossible", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=6.3, handlelength=1.1,
              handletextpad=0.5, borderpad=0.4, labelspacing=0.3, framealpha=0.9)
    save(fig, "fig1_mechanism")


# ---------------------------------------------------------------- Fig 2
def fig_collapse():
    # Plot BOTH swept arms. Plotting only xxh64-ints showed 40 of the 80 cells the
    # text and caption describe, and quoted the 80-cell RMS (1.96) over a 40-cell
    # picture whose own RMS is 2.35 -- the tighter ideal-hash arm was carrying the
    # number invisibly. Real hashing is filled, the ideal-hash control is open.
    cells = load("exp4_pooled_ci.json")
    arms = ["xxh64-ints", "ideal"]
    ms = sorted({c["m"] for c in cells if c["arm"] == arms[0]})
    sig = {(a, m): next(c["rmse_raw"] for c in cells
                        if c["arm"] == a and c["m"] == m and abs(c["d"] - 1.2) < 1e-9)
           for a in arms for m in ms}
    markers = {256: "o", 1024: "s", 4096: "^", 16384: "D"}
    cols = {256: C["blue"], 1024: C["green"], 4096: C["orange"], 16384: C["purple"]}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    aa = np.linspace(0, 3.6, 300)
    # the L-only baseline: reporting the counter alone has relative error d-1 = a*sigma,
    # so its "gain" over the raw sketch is exactly 100(1-a) -- one line for every m.
    ax.plot(aa, 100 * (1 - aa), color=C["gray"], lw=1.2, ls=":", zorder=4,
            label=r"report $L$ alone: $100(1-a)$")
    ax.plot(aa, gain(aa), color="k", lw=1.4, zorder=5,
            label=r"theory $1-\sqrt{g(a)}$")
    for a in arms:
        real = (a == "xxh64-ints")
        for m in ms:
            pts = [((c["d"] - 1) / sig[(a, m)], c["dclip"]) for c in cells
                   if c["arm"] == a and c["m"] == m]
            pts.sort()
            xs, ys = zip(*pts)
            if real:
                ax.scatter(xs, ys, s=22, marker=markers[m], color=cols[m],
                           edgecolor="white", linewidth=0.4, zorder=6, label=f"$m={m}$")
            else:
                ax.scatter(xs, ys, s=16, marker=markers[m], facecolor="none",
                           edgecolor=cols[m], linewidth=0.8, alpha=0.75, zorder=5,
                           label="ideal hash (control)" if m == ms[0] else None)
    ax.set_xlabel(r"standardized duplication $a=(d-1)/\sigma$")
    ax.set_ylabel("RMSE reduction (%)")
    ax.set_title("Every $(m,d)$ collapses onto one curve", fontsize=9.5)
    ax.set_xlim(-0.1, 3.6)
    ax.set_ylim(-12, 40)
    ax.axhline(0, color="0.75", lw=0.7, zorder=1)
    ax.annotate("$L$ alone is exact at $a{=}0$\nbut unbounded beyond",
                xy=(1.30, -6.5), fontsize=6.6, color=C["gray"], ha="left", va="center")
    ax.legend(fontsize=7.2, ncol=1, loc="upper right")
    save(fig, "fig2_collapse")


# ---------------------------------------------------------------- Fig 3
def fig_window():
    cells = load("exp4_pooled_ci.json")
    arm = "xxh64-ints"
    ms = sorted({c["m"] for c in cells if c["arm"] == arm})
    cols = {256: C["blue"], 1024: C["green"], 4096: C["orange"], 16384: C["purple"]}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for m in ms:
        pts = [(100 * (c["d"] - 1), c["dclip"]) for c in cells
               if c["arm"] == arm and c["m"] == m]
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.3, color=cols[m], label=f"$m={m}$")
    ax.axhline(5, color=C["gray"], ls=":", lw=1.0)
    ax.text(13, 6.2, "5% gain", fontsize=7.5, color=C["gray"])
    ax.set_xlabel("duplicate fraction $(d-1)$ (%)")
    ax.set_ylabel("RMSE reduction (%)")
    ax.set_title("The value window narrows as $m$ grows", fontsize=9.5)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-2, 36)
    ax.legend(fontsize=7.5)
    save(fig, "fig3_window")


# ---------------------------------------------------------------- Fig 4
def fig_universality():
    # one representative d=1.0 cell per family, with bootstrap CI where available
    bars = []  # (label, gain, lo, hi, color)

    e1 = load("exp1_pooled.json")
    real = [c for c in e1 if c["arm"] != "ideal-prng" and abs(c["d"] - 1.0) < 1e-9]
    g = np.mean([100 * (1 - c["rmse_clip"] / c["rmse_raw"]) for c in real])
    bars.append(("classic\nHLL", g, None, None, C["blue"]))

    e4 = load("exp4_pooled_ci.json")
    c = next(x for x in e4 if x["arm"] == "xxh64-ints" and x["m"] == 1024
             and abs(x["d"] - 1.0) < 1e-9)
    bars.append(("HLL++", c["dclip"], c["ci_clip"][0], c["ci_clip"][1], C["green"]))

    e3 = load("exp3_pooled_ci.json")
    for arm, lab, col in [("hll_hip", "HIP", C["orange"]), ("cpc", "CPC", C["purple"])]:
        cs = [x for x in e3 if x["arm"] == arm and abs(x["d"] - 1.0) < 1e-9]
        cc = next(x for x in cs if x["n"] == 8000)
        bars.append((lab, cc["dclip"], cc["ci_clip"][0], cc["ci_clip"][1], col))

    e6 = load("exp6_pooled_ci.json")
    cc = next(x for x in e6 if abs(x["d"] - 1.0) < 1e-9 and x["n"] == 8000)
    bars.append(("Ultra\nLogLog", cc["dclip"], cc["ci_clip"][0], cc["ci_clip"][1], C["red"]))

    e5 = load("exp5_pooled_ci.json")
    cs = [x for x in e5 if abs(x["d"] - 1.0) < 1e-9]
    g5 = np.mean([x["dclip"] for x in cs])
    lo5 = np.mean([x["ci_clip"][0] for x in cs])
    hi5 = np.mean([x["ci_clip"][1] for x in cs])
    bars.append(("PyPI\n(data)", g5, lo5, hi5, C["gray"]))

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    xs = np.arange(len(bars))
    for i, (lab, g, lo, hi, col) in enumerate(bars):
        ax.bar(i, g, color=col, alpha=0.85, width=0.7)
        if lo is not None:
            ax.errorbar(i, g, yerr=[[g - lo], [hi - g]], color="k", lw=1.0, capsize=2)
    ax.axhline(29.29, color="k", ls="--", lw=1.0)
    ax.text(len(bars) - 0.5, 34.0, r"theory $\approx 29.3\%$", fontsize=7.5,
            ha="right", color="k")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("RMSE reduction at $d{=}1$ (%)")
    ax.set_title("Gain is universal across sketch families", fontsize=9.5)
    ax.set_ylim(0, 38)
    save(fig, "fig4_universality")


# ---------------------------------------------------------------- Fig 5
def fig_secondorder():
    with open(os.path.join(PH1, "skew_theory.json")) as f:
        d = json.load(f)
    cells = d["cells"]
    meas = np.array([c["meas"] for c in cells])
    g1 = np.array([c["g1"] for c in cells])
    g3 = np.array([c["g3"] for c in cells])
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    lim = (-3, 36)
    ax.plot(lim, lim, color="k", lw=0.8, ls="--", zorder=1)
    ax.scatter(g1, meas, s=18, marker="o", color=C["gray"], alpha=0.7,
               edgecolor="white", linewidth=0.3,
               label=f"1st-order Gaussian (RMS {d['rms_g1']:.2f} pp)")
    ax.scatter(g3, meas, s=20, marker="D", color=C["red"], alpha=0.85,
               edgecolor="white", linewidth=0.3,
               label=f"+ bias, skew (RMS {d['rms_g3']:.2f} pp)")
    ax.set_xlabel("predicted RMSE reduction (%)")
    ax.set_ylabel("measured RMSE reduction (%)")
    ax.set_title("Three moments determine the gain", fontsize=9.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig5_secondorder")


# ---------------------------------------------------------------- Fig 6
def fig_competitors():
    with open(os.path.join(PH1, "exp9_competitors.json")) as f:
        cells = json.load(f)["cells"]
    n = 32000
    sel = [c for c in cells if c["n"] == n and c["d"] == 1.0]
    sel.sort(key=lambda c: c["m"])
    ms = [c["m"] for c in sel]
    clip = [c["rmse_hll_clip"] for c in sel]
    gee = [c["rmse_gee"] for c in sel]
    x = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.bar(x - 0.2, gee, width=0.4, color=C["gray"], alpha=0.85, label="GEE sampling")
    ax.bar(x + 0.2, clip, width=0.4, color=C["blue"], alpha=0.9, label="HLL + clip")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$2^{{{int(np.log2(m))}}}$" for m in ms])
    ax.set_xlabel("memory budget: $m$ registers,\nor a sample of the same size")
    ax.set_ylabel("relative RMSE (log)")
    ax.set_title("Using $L$ by sampling is far worse", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=True, borderaxespad=0.3)
    for xi, g, c in zip(x, gee, clip):
        ax.text(xi, max(g, c) * 1.4, f"{g/c:.0f}$\\times$", ha="center", fontsize=7)
    ax.set_ylim(1e-3, 3)
    save(fig, "fig6_competitors")


# ---------------------------------------------------------------- Fig 7
def fig_dbcolumns():
    """Database columns placed on the closed-form value window.

    The same collapse argument as Fig. 2, but every point is a column of a real
    table rather than a swept synthetic cell. Identity is carried by marker shape
    as well as hue, so the figure survives greyscale printing and colour-vision
    deficiency. Three low-cardinality columns (NDV <= 50) are estimated exactly by
    the sketch, leaving sigma ~ 0 and a undefined; they are omitted and noted in
    the caption.
    """
    with open(os.path.join(PH1, "exp13_ndv_columns.json")) as f:
        cells = json.load(f)["cells"]
    style = {"nasa": ("NASA-HTTP", C["blue"], "s"),
             "pypi": ("PyPI", C["orange"], "D"),
             "tpch": ("TPC-H", C["green"], "o")}

    keyed, other, exact = [], [], 0
    for c in cells:
        sig = c["rmse_raw"]
        if sig < 1e-6:                      # sketch exact: a is undefined
            exact += 1
            continue
        a = (c["d_mean"] - 1) / sig
        (keyed if a < 1e-9 else other).append((c["table"], a, c["gain_pct"]))

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    aa = np.concatenate([np.linspace(0, 1, 200), np.logspace(0, 4.3, 300)])
    ax.plot(aa, gain(aa), color="k", lw=1.4, zorder=3,
            label=r"theory $1-\sqrt{g(a)}$")
    ax.axhline(0, color="0.75", lw=0.7, zorder=1)

    # the key columns all sit at a=0; separate them horizontally so each is visible
    seen = set()
    for i, (tbl, a, g) in enumerate(sorted(keyed, key=lambda t: -t[2])):
        lab, colr, mk = style[tbl]
        ax.scatter(i * 0.055, g, s=46, marker=mk, color=colr, zorder=6,
                   edgecolor="white", linewidth=0.9,
                   label=lab if lab not in seen else None)
        seen.add(lab)
    for tbl, a, g in other:
        lab, colr, mk = style[tbl]
        ax.scatter(a, g, s=46, marker=mk, color=colr, zorder=6,
                   edgecolor="white", linewidth=0.9,
                   label=lab if lab not in seen else None)
        seen.add(lab)

    # real key values carried through the descent by adding controlled duplicate rows
    swp = os.path.join(PH1, "exp15_ndv_sweep.json")
    if os.path.exists(swp):
        with open(swp) as f:
            rows = json.load(f)["cells"]
        for col in sorted({r["column"] for r in rows}):
            pts = sorted([r for r in rows if r["column"] == col], key=lambda r: r["a"])
            tbl = col.split(".")[0]
            _, colr, mk = style[tbl]
            x = [r["a"] for r in pts]
            y = [r["gain_pct"] for r in pts]
            ax.plot(x, y, color=colr, lw=1.0, ls="--", alpha=0.75, zorder=4)
            # (bootstrap CIs are reported per point in the text, not drawn here)
            ax.scatter(x, y, s=22, marker=mk, facecolor="none", edgecolor=colr,
                       linewidth=1.1, zorder=5,
                       label="same keys, added duplicates"
                       if col == sorted({r["column"] for r in rows})[0] else None)

    # (the four key columns at a=0 are named in the caption; no label needed here)
    ax.annotate(f"{len(other)} non-key columns,\nall exactly $0.0\\%$",
                xy=(90, 3.4), fontsize=6.6, color=C["gray"], ha="left")

    ax.set_xscale("symlog", linthresh=1, linscale=0.9)
    ax.set_xlim(-0.12, 4e4)
    ax.set_ylim(-3, 38)
    ax.set_xlabel(r"standardized duplication $a=(d-1)/\sigma$")
    ax.set_ylabel("RMSE reduction (%)")
    ax.set_title("Database columns fall on the same curve", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="upper right", framealpha=0.9)
    save(fig, "fig7_dbcolumns")


if __name__ == "__main__":
    fig_mechanism()
    fig_collapse()
    fig_window()
    fig_universality()
    fig_secondorder()
    fig_competitors()
    fig_dbcolumns()
    print("all figures written to", FIG)

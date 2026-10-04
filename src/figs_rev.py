"""figs_rev.py: revision figures (larger labels; architecture; trigger budget; missed-trigger bands)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, pandas as pd
import plotting
from config import FIGS, TABLES
RC = {"font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes"], "mathtext.fontset": "stix",
      "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
      "legend.fontsize": 9.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(RC)

def save(fig, n):
    fig.savefig(FIGS / f"{n}.pdf", bbox_inches="tight"); fig.savefig(FIGS / f"{n}.png", dpi=250, bbox_inches="tight"); plt.close(fig)

def box(ax, x, y, w, h, t, c, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=c, ec="k", lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.15)

def arr(ax, p, q, ls="-", c="k"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, ls=ls, color=c, shrinkA=1, shrinkB=1))

def fig_architecture():
    W, H = 7.2, 4.45
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    box(ax, 0.05, 3.05, 1.55, 0.8, "Rain gauge\n(tipping bucket)", "#deebf7")
    box(ax, 0.05, 1.95, 1.55, 0.8, "Irrigation\ncontroller (valve)", "#deebf7")
    box(ax, 2.0, 2.45, 1.45, 0.9, "Trigger\n$g(t)$ = rain $\\geq\\theta$\nor flow $>0$", "#fff7bc")
    arr(ax, (1.6, 3.4), (2.0, 3.05)); arr(ax, (1.6, 2.35), (2.0, 2.75))
    opts = [(3.25, "(a) Wired interrupt\nto co-located node\n$\\approx$0 $\\mu$A, $\\leq$1 step", "#e5f5e0"),
            (2.0, "(b) Wake-up receiver\nlistening current $I_w$\nlatency: ms", "#fdd0a2"),
            (0.75, "(c) Gateway, LoRaWAN\nClass B ping slots\n21.5 $\\mu$A (128 s), $\\leq$128 s", "#fde0dd")]
    for y, t, c in opts:
        box(ax, 3.85, y, 1.65, 0.95, t, c, fs=9.5)
        arr(ax, (3.45, 2.9), (3.85, y + 0.47))
    box(ax, 5.85, 1.55, 1.3, 1.6, "Sensor node\nsleep $\\to$ wake:\nquiet $T_q$\nevent $T_e$\nhold $H$", "#f0f0f0", fs=9.5)
    for y, _, _ in opts:
        arr(ax, (5.5, y + 0.47), (5.85, 2.35))
    ax.text(3.6, 0.25, "Energy parity with fixed sampling of equal accuracy requires an average trigger current\nbelow 0.7 to 2.5 $\\mu$A (SF7 to SF9): satisfied by (a), by (b) only for sub-$\\mu$A receivers, never by (c).",
            ha="center", va="center", fontsize=9.5, style="italic")
    save(fig, "figR1_architecture")

def fig_budget():
    B = pd.read_csv(TABLES / "table_rev_trigger_budget.csv")
    B = B[(B.probe_ms == 10) & (B.sleep_uA == B.sleep_uA.min())].sort_values("SF")
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(3)
    ax.bar(x - 0.2, B.trigger_budget_uA.values, 0.4, color="#2ca02c", label="trigger budget $I^*$")
    ax.bar(x + 0.2, B.classB128_uA.values, 0.4, color="#d62728", label="Class B, 128 s pings")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"SF{s}" for s in B.SF])
    ax.set_ylabel("average current ($\\mu$A)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1); ax.grid(axis="y", alpha=0.3)
    save(fig, "figR2_trigger_budget")

def fig_missed():
    S = pd.read_csv(TABLES / "table_rev_missed_bands.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, m, lab, ref in [(axes[0], "rmse_interp", "interpolation RMSE (m$^3$/m$^3$)", (0.00336, 0.00198)),
                            (axes[1], "delay_mean", "mean detection delay (min)", (116.8, 51.7))]:
        p = S.p_miss.values
        ax.fill_between(p, S[f"{m}_2.5"], S[f"{m}_97.5"], color="#9ecae1", alpha=0.6, label="95% band (100 draws)")
        ax.plot(p, S[f"{m}_50"], "o-", color="#08519c", label="event-aware, median")
        ax.axhline(ref[0], color="#7f7f7f", ls="--", label="fixed 4 h"); ax.axhline(ref[1], color="#7f7f7f", ls=":", label="fixed 2 h")
        ax.set_xlabel("share of trigger episodes missed"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); save(fig, "figR3_missed")

if __name__ == "__main__":
    plt.rcParams.update(RC)
    # regenerate the original figures with larger labels
    import exp_main
    d = exp_main.load()
    for f in [plotting.fig1_data, plotting.fig3_example, plotting.fig5_where_gains]:
        plt.rcParams.update(RC); f(d)
    for f in [plotting.fig2_pareto, plotting.fig4_delays, plotting.fig6_sensitivity]:
        plt.rcParams.update(RC); f()
    fig_architecture(); fig_budget(); fig_missed()
    print("ok")

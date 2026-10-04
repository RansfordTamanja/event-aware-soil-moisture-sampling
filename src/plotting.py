"""
plotting.py

All figures, from saved results and the aligned data only.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA, FIGS, PORTS, TABLES
from evaluate import DAY_STEPS
from exp_main import FAMILIES, MAIN_RATE, load, run_policy, split_masks

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
COL = {"Fixed": "#7f7f7f", "Fixed-120": "#444444", "Reactive": "#1f77b4", "Event-aware": "#d62728",
       "Event-aware + reactive": "#ff7f0e", "Oracle SoD": "black"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(" ", name)


def fig1_data(d):
    cal, _ = split_masks(len(d))
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    for ax, ports, lab in [(axes[0], ["Port1", "Port3", "Port5"], "30 cm"), (axes[1], ["Port2", "Port4", "Port6"], "60 cm")]:
        for p in ports:
            ax.plot(d.index, d[p], lw=0.8, label=p)
        ax.set_ylabel(f"VWC, {lab} (m³/m³)")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
    daily = d.rain_mm.resample("D").sum()
    axes[2].bar(daily.index, daily.values, width=1, color="#1f77b4", label="rain (mm/day)")
    irr = (d.irr_lps > 0).resample("D").mean() * daily.max()
    axes[2].fill_between(irr.index, 0, irr.values, step="mid", color="#2ca02c", alpha=0.3, label="irrigation (share of day, scaled)")
    axes[2].set_ylabel("mm / day"); axes[2].legend(fontsize=7, loc="upper left")
    days = pd.Series(cal, index=d.index).resample("D").max()
    for ax in axes:
        for t, c in days.items():
            if c:
                ax.axvspan(t, t + pd.Timedelta(days=1), color="grey", alpha=0.07, lw=0)
    save(fig, "fig1_study_data")


def fig2_pareto():
    g = json.load(open(DATA / "grid_results.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for fam in FAMILIES + ["Oracle SoD"]:
        for ax, m in zip(axes, ["rmse_interp", "rmse_hold"]):
            pts = sorted((c["test"]["rate"], c["test"][m]) for c in g[fam])
            env, best = [], np.inf
            for r, e in pts:
                if e < best and e > 0:
                    env.append((r, e)); best = e
            vis = [p for p in pts if p[0] < 0.999 and p[1] > 0]
            ax.scatter([p[0] for p in vis], [p[1] for p in vis], s=6, color=COL[fam], alpha=0.25)
            ax.plot([p[0] for p in env], [p[1] for p in env], "o-" if fam != "Oracle SoD" else "k--",
                    ms=3, color=COL[fam], label=("Send-on-delta reference (unrealisable)" if fam == "Oracle SoD" else fam))
    for ax, t in zip(axes, ["Interpolated reconstruction", "Real-time (held) value"]):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sampling rate (share of native samples)"); ax.set_ylabel("RMSE (m$^3$/m$^3$)"); ax.set_title(t)
    axes[0].legend(fontsize=8.5, frameon=False)
    save(fig, "fig2_rate_error_tradeoff")


def fig3_example(d):
    sel = json.load(open(DATA / "selected.json"))
    g = d.trigger.values
    x = d["Port3"].values
    # choose the test-week window around the largest 2-day rise of Port3
    _, test_m = split_masks(len(d))
    rise = pd.Series(x).diff(8).where(test_m, np.nan)
    c = int(rise.idxmax())
    lo, hi = max(0, c - DAY_STEPS), min(len(x), c + 2 * DAY_STEPS)
    t = d.index[lo:hi]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    act = g[lo:hi] > 0
    for i in np.flatnonzero(act):
        ax.axvspan(t[i], t[i] + pd.Timedelta(minutes=15), color="#2ca02c", alpha=0.15, lw=0)
    ax.plot(t, x[lo:hi], color="black", lw=1.2, label="15-min record (Port 3, 30 cm)")
    for fam, off in [("Fixed-120", 0), ("Reactive", 0), ("Event-aware", 0)]:
        cfg = {"T": 120} if fam == "Fixed-120" else sel[fam][str(MAIN_RATE)]
        S = run_policy("Fixed" if fam == "Fixed-120" else fam, cfg, x, g)
        S = S[(S >= lo) & (S < hi)]
        ax.plot(d.index[S], x[S], "o-", ms=3.5, lw=0.8, color=COL[fam], alpha=0.85, label=f"{fam} samples")
    ax.set_ylabel("VWC (m³/m³)"); ax.legend(fontsize=7)
    ax.set_title("A wetting event (green: rain or irrigation trigger active)")
    save(fig, "fig3_event_example")


def fig4_delays():
    J = json.load(open(DATA / "day_errors.json"))
    ev = J["event_delays"]
    names = [n for n in ["Fixed", "Fixed-120", "Reactive", "Event-aware"] if n in ev]
    data = [np.concatenate([ev[n][p] for p in PORTS]) for n in names]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    disp = {"Fixed": "Fixed 4 h", "Fixed-120": "Fixed 2 h", "Reactive": "Reactive", "Event-aware": "Event-aware"}
    ax.boxplot(data, labels=[disp[n] for n in names], showfliers=True)
    ax.set_ylabel("extra detection delay (min)")
    save(fig, "fig4_detection_delay")


def fig5_where_gains(d):
    """Daily errors are grouped in 24-h blocks counted from the first record,
    so triggers are aggregated over the same blocks."""
    J = json.load(open(DATA / "day_errors.json"))
    de = J["day_err"]
    _, test_m = split_masks(len(d))
    block = np.arange(len(d)) // DAY_STEPS
    trig_block = pd.Series(d.trigger.values).groupby(block).max()
    tb = np.unique(block[test_m])
    ev = (trig_block.reindex(tb).values > 0) | (trig_block.reindex(tb - 1).fillna(0).values > 0)
    a = np.array(de["Event-aware"]["interp"]); b = np.array(de["Fixed-120"]["interp"])
    diff = (a - b) * 1e6
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(np.arange(len(diff)), diff, color=np.where(ev, "#2ca02c", "#999999"), width=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("test day"); ax.set_ylabel("daily MSE difference (×10⁻⁶)")
    ax.set_title("Event-aware minus fixed 120-min (green: trigger on the day or the day before)", fontsize=8.5)
    save(fig, "fig5_where_gains_arise")


def fig6_sensitivity():
    t = pd.read_csv(TABLES / "table_sens_triggers.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    src = t[t.hold_min.astype(str) == "grid"]
    axes[0].barh(src.variant, src["rmse_interp@0.1"] * 1e3, color=["#d62728"] + ["#999999"] * (len(src) - 1))
    axes[0].set_xlabel("RMSE at 10 % sampling (×10⁻³ m³/m³)"); axes[0].set_title("Trigger source")
    h = t[t.hold_min.astype(str) != "grid"].copy()
    h["hold_min"] = h.hold_min.astype(int)
    h = h.sort_values("hold_min")
    axes[1].plot(h.hold_min, h["rmse_interp@0.1"] * 1e3, "o-", color="#d62728", label="10 % sampling")
    axes[1].plot(h.hold_min, h["rmse_interp@0.2"] * 1e3, "s--", color="#d62728", alpha=0.6, label="20 % sampling")
    axes[1].set_xlabel("hold after trigger (min)"); axes[1].set_ylabel("RMSE (×10⁻³ m³/m³)")
    axes[1].set_title("Hold period"); axes[1].legend(fontsize=7)
    save(fig, "fig6_trigger_sensitivity")


def fig7_trigger_realism():
    lt = pd.read_csv(TABLES / "table_latency.csv")
    mt = pd.read_csv(TABLES / "table_missed.csv")
    ot = pd.read_csv(TABLES / "table_overhead.csv")
    env = pd.read_csv(TABLES / "table_envelope.csv").set_index("rate")
    fixed10 = env.loc[0.10, "rmse_interp[Fixed]"] * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    axes[0].plot(lt.latency_min, lt["env_rmse_interp@0.1"] * 1e3, "o-", color="#d62728", label="event-aware")
    axes[0].axhline(fixed10, color="#7f7f7f", ls="--", label="fixed, same rate")
    axes[0].set_xlabel("trigger latency (min)"); axes[0].set_ylabel("RMSE at 10 % (×10⁻³)")
    axes[0].set_title("Trigger latency"); axes[0].legend(fontsize=7)
    axes[1].plot(mt.p_miss * 100, mt.rmse_interp * 1e3, "o-", color="#d62728", label="event-aware (mean)")
    axes[1].fill_between(mt.p_miss * 100, mt.rmse_interp * 1e3, mt.rmse_interp_p95 * 1e3, color="#d62728", alpha=0.15,
                         label="95th percentile")
    axes[1].axhline(fixed10, color="#7f7f7f", ls="--", label="fixed, 10 %")
    axes[1].set_xlabel("missed trigger episodes (%)"); axes[1].set_ylabel("RMSE (×10⁻³)")
    axes[1].set_title("Missed triggers"); axes[1].legend(fontsize=7)
    psis = [c for c in ot.columns if c.startswith("fixed_rmse@psi=")]
    x = [float(c.split("=")[1]) * 100 for c in psis]
    for _, r in ot.iterrows():
        if r.event_aware_rate in (0.1, 0.2):
            l, = axes[2].plot(x, [r[c] * 1e3 for c in psis], "o--", ms=3, label=f"fixed, same energy as EA {r.event_aware_rate:.0%}")
            axes[2].axhline(r.event_aware_rmse * 1e3, color=l.get_color(), lw=1, label=f"event-aware {r.event_aware_rate:.0%}")
    axes[2].set_xlabel("overhead ψ (% of native energy)"); axes[2].set_ylabel("RMSE (×10⁻³)")
    axes[2].set_title("Equal-energy comparison"); axes[2].legend(fontsize=6)
    save(fig, "fig7_trigger_realism")


def main():
    d = load()
    fig1_data(d); fig2_pareto(); fig3_example(d); fig4_delays(); fig5_where_gains(d); fig6_sensitivity()
    fig7_trigger_realism()


if __name__ == "__main__":
    main()

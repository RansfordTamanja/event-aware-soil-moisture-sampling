"""
exp_rev2.py

(1) Missed-trigger Monte Carlo with 100 draws per miss probability; per-draw results and
    2.5/50/97.5 percentiles of interpolation RMSE, hold RMSE, and mean detection delay.
(2) Trigger energy budget: the largest average trigger-channel current I* for which
    event-aware sampling at the 10 percent operating point (test rate 0.106) uses no more
    energy than fixed sampling at the rate needed to match its interpolation error
    (fixed_rate_for_same_error from table_overhead, rate 0.10 row), for each radio
    setting, probe time and sleep floor of energy_hw.py.
"""
import json
import numpy as np, pandas as pd
from config import *
from exp_main import load, split_masks, make_trigger, evaluate_config
from exp_trigger import episodes
from prepare_data import wetting_onsets
import energy_hw as hw

def main():
    d = load(); _, test_m = split_masks(len(d))
    ons = {p: wetting_onsets(d[p].values) for p in PORTS}
    sel = json.load(open(DATA / "selected.json"))["Event-aware"]["0.1"]
    g0 = make_trigger(d, rain_thr=sel["rain_thr"]); eps = episodes(g0)
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for pm in [0.0, 0.1, 0.25, 0.5, 0.75]:
        for s in range(100 if pm > 0 else 1):
            g = g0.copy()
            for a, b in eps:
                if rng.random() < pm:
                    g[a:b + 1] = False
            r = evaluate_config(d, "Event-aware", sel, test_m, ons, g=g)
            rows.append({"p_miss": pm, "draw": s, "rate": r["rate"], "rmse_interp": r["rmse_interp"],
                         "rmse_hold": r["rmse_hold"], "delay_mean": r["delay_mean_min"]})
    D = pd.DataFrame(rows); D.to_csv(TABLES / "table_rev_missed_draws.csv", index=False)
    S = D.groupby("p_miss").agg(**{f"{m}_{q}": (m, (lambda v, q=q: np.percentile(v, q))) for m in ["rate", "rmse_interp", "rmse_hold", "delay_mean"] for q in (2.5, 50, 97.5)}).reset_index()
    S.to_csv(TABLES / "table_rev_missed_bands.csv", index=False); print(S.round(5).to_string(index=False))
    ov = pd.read_csv(TABLES / "table_overhead.csv"); rf = float(ov.loc[(ov.event_aware_rate - 0.10).abs().idxmin(), "fixed_rate_for_same_error"])
    out = []
    for sf in [7, 9, 12]:
        for tp in [0.010, 0.050]:
            for isl in [hw.I_SLEEP_CHIP, 10e-6, 50e-6]:
                qs = hw.charge_per_sample(sf, tp)
                q_ea = 0.106 * 96 * qs; q_fx = rf * 96 * qs
                istar = (q_fx - q_ea) / 86400
                out.append({"SF": sf, "probe_ms": tp * 1e3, "sleep_uA": round(isl * 1e6, 2), "fixed_rate_same_error": rf,
                            "EA_sampling_uA": q_ea / 86400 * 1e6, "fixed_sampling_uA": q_fx / 86400 * 1e6,
                            "trigger_budget_uA": istar * 1e6,
                            "classB128_uA": hw.classb_charge_per_day(128) / 86400 * 1e6})
    B = pd.DataFrame(out); B.to_csv(TABLES / "table_rev_trigger_budget.csv", index=False)
    (TABLES / "table_rev_trigger_budget.md").write_text(B.to_markdown(index=False, floatfmt=".4g"))
    print(B[B.sleep_uA == B.sleep_uA.min()].round(3).to_string(index=False))

if __name__ == "__main__":
    main()

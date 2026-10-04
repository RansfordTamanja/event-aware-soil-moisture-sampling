"""
exp_trigger.py

Experiment 3: relaxing the ideal-trigger assumption.

The main experiments assume an ideal trigger: the node learns of rain or
irrigation in the same 15-min step in which it is recorded, and no trigger is
missed. This script quantifies what happens when that is not so.

  (a) Latency. Every trigger reaches the node L minutes late
      (L in LATENCY_SWEEP), covering gauge reporting, backhaul and downlink
      delays. The configuration selected at the main rate is re-scored, and the
      whole event-aware grid is rerun to give the equal-rate envelope.
  (b) Missed triggers. Each trigger episode (consecutive active steps, merged
      across gaps of up to one hour) is dropped with probability p
      (MISS_SWEEP), over N_MISS_SEEDS random seeds, as would happen with an
      unreliable downlink or a gauge that misses local showers.
  (c) Energy of receiving triggers. Listening for triggers (LoRaWAN Class B
      ping slots, a wake-up receiver or a wired interrupt) adds an overhead psi,
      expressed as a share of the native schedule's daily energy. At equal total
      energy, event-aware sampling at rate r is compared with fixed sampling at
      r + psi / (1 - phi); the break-even overhead psi* is the value at which
      the two errors are equal.

Outputs: outputs/data/trigger.json, outputs/tables/table_latency.*,
         table_missed.*, table_overhead.*
"""

import json

import numpy as np
import pandas as pd

from config import (DATA, LATENCY_SWEEP, MISS_SWEEP, N_MISS_SEEDS, PHI_DEFAULT, PORTS, PSI_SWEEP,
                    SEED, TABLES)
from exp_main import (MAIN_RATE, configs, evaluate_config, load, make_trigger, split_masks)
from exp_sensitivity import envelope_at
from prepare_data import wetting_onsets


def episodes(g, merge_steps=4):
    idx = np.flatnonzero(g)
    if idx.size == 0:
        return []
    eps, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_steps:
            eps.append((start, prev)); start = i
        prev = i
    eps.append((start, prev))
    return eps


def main():
    d = load()
    onsets = {p: wetting_onsets(d[p].values) for p in PORTS}
    _, test_m = split_masks(len(d))
    sel = json.load(open(DATA / "selected.json"))["Event-aware"][str(MAIN_RATE)]
    out = {"selected": sel}

    # (a) latency
    lat_rows = []
    for L in LATENCY_SWEEP:
        g = make_trigger(d, rain_thr=sel["rain_thr"], latency_min=L)
        r = evaluate_config(d, "Event-aware", sel, test_m, onsets, g=g)
        res = []
        for c in configs("Event-aware"):
            gc = make_trigger(d, rain_thr=c["rain_thr"], latency_min=L)
            res.append(evaluate_config(d, "Event-aware", c, test_m, onsets, g=gc))
        lat_rows.append({"latency_min": L, "rate": r["rate"], "rmse_interp": r["rmse_interp"],
                         "rmse_hold": r["rmse_hold"], "delay_mean_min": r["delay_mean_min"],
                         "delay_p90_min": r["delay_p90_min"], "peak_err_mean": r["peak_err_mean"],
                         "env_rmse_interp@0.1": envelope_at([(x["rate"], x["rmse_interp"]) for x in res], 0.10),
                         "env_rmse_hold@0.1": envelope_at([(x["rate"], x["rmse_hold"]) for x in res], 0.10)})
        print(lat_rows[-1])
    lt = pd.DataFrame(lat_rows)
    lt.to_csv(TABLES / "table_latency.csv", index=False)
    (TABLES / "table_latency.md").write_text(lt.to_markdown(index=False, floatfmt=".5f"))

    # (b) missed triggers
    g0 = make_trigger(d, rain_thr=sel["rain_thr"])
    eps = episodes(g0)
    rng = np.random.default_rng(SEED)
    miss_rows = []
    for pm in MISS_SWEEP:
        vals = []
        for s in range(N_MISS_SEEDS if pm > 0 else 1):
            g = g0.copy()
            for (a, b) in eps:
                if rng.random() < pm:
                    g[a:b + 1] = False
            r = evaluate_config(d, "Event-aware", sel, test_m, onsets, g=g)
            vals.append((r["rate"], r["rmse_interp"], r["rmse_hold"], r["delay_mean_min"], r["recall"]))
        v = np.array(vals)
        miss_rows.append({"p_miss": pm, "episodes": len(eps), "rate": v[:, 0].mean(),
                          "rmse_interp": v[:, 1].mean(), "rmse_interp_p95": np.percentile(v[:, 1], 95),
                          "rmse_hold": v[:, 2].mean(), "delay_mean_min": v[:, 3].mean(),
                          "recall": v[:, 4].mean()})
        print(miss_rows[-1])
    mt = pd.DataFrame(miss_rows)
    mt.to_csv(TABLES / "table_missed.csv", index=False)
    (TABLES / "table_missed.md").write_text(mt.to_markdown(index=False, floatfmt=".5f"))

    # (c) trigger-reception overhead at equal total energy
    grid = json.load(open(DATA / "grid_results.json"))
    ea = [(c["test"]["rate"], c["test"]["rmse_interp"]) for c in grid["Event-aware"]]
    fx = [(c["test"]["rate"], c["test"]["rmse_interp"]) for c in grid["Fixed"]]
    fxs = sorted(fx)
    ov_rows = []
    for r in (0.075, 0.10, 0.15, 0.20):
        e_ea = envelope_at(ea, r)
        row = {"event_aware_rate": r, "event_aware_rmse": e_ea}
        for psi in PSI_SWEEP:
            rf = r + psi / (1 - PHI_DEFAULT)
            row[f"fixed_rmse@psi={psi}"] = envelope_at(fx, rf)
        # break-even: fixed rate needed to match e_ea, from the fixed envelope (log-log)
        rr = np.array([p[0] for p in fxs]); ee = np.array([p[1] for p in fxs])
        ok = ee > 0
        rr, ee = rr[ok], ee[ok]
        order = np.argsort(ee)
        rf_star = float(np.exp(np.interp(np.log(e_ea), np.log(ee[order]), np.log(rr[order]))))
        # if event-aware error is below every fixed configuration with rate < 1, the
        # break-even is only bounded below by the largest such fixed rate
        censored = bool(e_ea < ee.min())
        row["fixed_rate_for_same_error"] = rf_star
        row["break_even_psi"] = (rf_star - r) * (1 - PHI_DEFAULT)
        row["break_even_is_lower_bound"] = censored
        ov_rows.append(row)
        print(row)
    ot = pd.DataFrame(ov_rows)
    ot.to_csv(TABLES / "table_overhead.csv", index=False)
    (TABLES / "table_overhead.md").write_text(ot.to_markdown(index=False, floatfmt=".5f"))
    with open(DATA / "trigger.json", "w") as f:
        json.dump({"selected": sel, "latency": lat_rows, "missed": miss_rows, "overhead": ov_rows,
                   "episodes": len(eps)}, f, indent=1, default=float)


if __name__ == "__main__":
    main()

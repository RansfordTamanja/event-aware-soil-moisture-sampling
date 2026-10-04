"""
exp_sensitivity.py

Experiment 2: sensitivity of event-aware sampling to its trigger design, and
of the energy saving to the energy model.

  (a) trigger source: rain and irrigation, rain only, irrigation only
  (b) hold period after a trigger: each value of HOLD_SWEEP on its own
For each variant the event-aware grid (hold periods and rain thresholds
included) is rerun and the test-week lower envelope is interpolated at
sampling rates 0.10 and 0.20. In the main experiments the hold period and the
rain threshold are selected on the calibration weeks.

  (d) energy: relative daily energy E = phi + (1 - phi) * rate and the
      implied lifetime multiplier 1/E, for the configurations selected on the
      calibration weeks at the main rate, over phi in PHI_SWEEP.

Outputs: outputs/tables/table_sens_triggers.*, table_energy.*
"""

import json

import numpy as np
import pandas as pd

from config import HOLD_SWEEP, PHI_SWEEP, PORTS, TABLES
from exp_main import configs, evaluate_config, load, make_trigger, split_masks
from prepare_data import wetting_onsets


def envelope_at(points, R):
    pts = sorted(points)
    env, best = [], np.inf
    for r_, e_ in pts:
        if e_ < best:
            env.append((r_, e_)); best = e_
    rr = np.array([p[0] for p in env]); ee = np.array([p[1] for p in env])
    if rr.size == 0 or not (rr.min() <= R <= rr.max()):
        return np.nan
    return float(np.exp(np.interp(np.log(R), np.log(rr), np.log(ee))))


def variant(d, onsets, test_m, use_rain=True, use_irr=True, holds=None):
    """Rerun the event-aware grid with the given trigger sources (and optionally
    only some hold periods) and interpolate the test-week envelope."""
    res = []
    for c in configs("Event-aware"):
        if holds is not None and c["hold"] not in holds:
            continue
        g = make_trigger(d, rain_thr=c["rain_thr"], use_rain=use_rain, use_irr=use_irr)
        res.append(evaluate_config(d, "Event-aware", c, test_m, onsets, g=g))
    out = {}
    for R in (0.10, 0.20):
        out[f"rmse_interp@{R}"] = envelope_at([(r["rate"], r["rmse_interp"]) for r in res], R)
        out[f"rmse_hold@{R}"] = envelope_at([(r["rate"], r["rmse_hold"]) for r in res], R)
    out["min_rate"] = min(r["rate"] for r in res)
    return out


def main():
    d = load()
    onsets = {p: wetting_onsets(d[p].values) for p in PORTS}
    _, test_m = split_masks(len(d))
    rows = []
    for name, ur, ui in [("Rain and irrigation", True, True), ("Rain only", True, False),
                         ("Irrigation only", False, True)]:
        rows.append({"variant": name, "hold_min": "grid", **variant(d, onsets, test_m, ur, ui)})
        print(rows[-1])
    for hold in HOLD_SWEEP:
        rows.append({"variant": "Rain and irrigation", "hold_min": hold,
                     **variant(d, onsets, test_m, holds={hold})})
        print(rows[-1])
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table_sens_triggers.csv", index=False)
    (TABLES / "table_sens_triggers.md").write_text(t.to_markdown(index=False, floatfmt=".5f"))

    main_tab = pd.read_csv(TABLES / "table_main.csv")
    er = []
    for _, r in main_tab.iterrows():
        for phi in PHI_SWEEP:
            e = phi + (1 - phi) * r["rate"]
            er.append({"policy": r["policy"], "rate": r["rate"], "phi": phi, "energy_rel": e,
                       "lifetime_multiplier": 1 / e})
    et = pd.DataFrame(er)
    et.to_csv(TABLES / "table_energy.csv", index=False)
    (TABLES / "table_energy.md").write_text(et.to_markdown(index=False, floatfmt=".3f"))
    print(et.pivot(index="phi", columns="policy", values="lifetime_multiplier").round(2).to_string())


if __name__ == "__main__":
    main()

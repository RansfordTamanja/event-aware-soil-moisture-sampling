"""
exp_seasons.py

Experiment 4: out-of-period validation on seasons outside the study window.

The soil and rainfall records start on 30 June 2021, before the irrigation
record (7 April 2022). Two further periods are complete:
  Summer-autumn 2021   2021-07-01 to 2021-10-31  (rain triggers only; the grove
                       was irrigated but the irrigation record does not cover it)
  Winter 2021-22       2021-12-01 to 2022-04-06  (rain triggers only; no
                       irrigation season)
The configurations selected on the 2022 calibration weeks are applied
unchanged (no re-tuning) with rain-only triggers, and all policies are scored
over the whole of each period. The share of wetting onsets preceded by a rain
trigger within 6 h shows how many events a rain-only trigger can explain.

Outputs: outputs/data/seasons.json, outputs/tables/table_seasons.*
"""

import json

import numpy as np
import pandas as pd

from config import DATA, PORTS, RAIN_VALID_FLAGS, STEP_MIN, TABLES, TARGET_RATES
from exp_main import FAMILIES, aggregate, run_policy, score
from prepare_data import load as load_raw, wetting_onsets

PERIODS = {"Summer-autumn 2021": ("2021-07-01", "2021-10-31 23:45"),
           "Winter 2021-22": ("2021-12-01", "2022-04-06 23:45")}
RATES = (0.10, 0.20)


def build(start, end):
    s, rain, _, _ = load_raw()
    grid = pd.date_range(start, end, freq=f"{STEP_MIN}min")
    df = s.reindex(grid).interpolate(limit=4, limit_area="inside")
    df["rain_mm"] = rain.reindex(grid).fillna(0.0)
    df["irr_lps"] = 0.0
    return df


def main():
    sel = json.load(open(DATA / "selected.json"))
    rows, info = [], {}
    for name, (a, b) in PERIODS.items():
        d = build(a, b)
        missing = int(d[PORTS].isna().any(axis=1).sum())
        d = d.ffill().bfill()
        mask = np.ones(len(d), bool)
        onsets = {p: wetting_onsets(d[p].values) for p in PORTS}
        n_on = sum(len(v) for v in onsets.values())
        info[name] = {"days": len(d) / 96, "missing_steps_filled": missing, "rain_mm": float(d.rain_mm.sum()),
                      "onsets": n_on}
        for R in RATES:
            for fam in FAMILIES + ["Fixed-120"]:
                cfg = sel.get(fam, {}).get(str(R))
                if fam == "Fixed-120":
                    cfg = {"T": 120}
                if cfg is None:
                    continue
                thr = cfg.get("rain_thr", 0.2)
                g = d.rain_mm.values >= thr
                per = []
                for p in PORTS:
                    S = run_policy("Fixed" if fam == "Fixed-120" else fam, cfg, d[p].values, g)
                    per.append(score(d[p].values, S, mask, onsets=onsets[p]))
                agg = aggregate(per)
                rows.append({"period": name, "target_rate": R, "policy": fam,
                             **{k: agg[k] for k in ["rate", "rmse_interp", "rmse_hold", "maxerr_hold",
                                                    "recall", "delay_mean_min", "events"]}})
        # how many onsets can a rain-only trigger explain?
        g = d.rain_mm.values >= 0.2
        explained = 0
        for p in PORTS:
            for o in onsets[p]:
                if g[max(0, o - 24):o + 9].any():
                    explained += 1
        info[name]["onsets_with_rain_within_6h"] = explained
        print(name, info[name])
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    t.to_csv(TABLES / "table_seasons.csv", index=False)
    (TABLES / "table_seasons.md").write_text(t.to_markdown(index=False, floatfmt=".4f"))
    with open(DATA / "seasons.json", "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=1, default=float)


if __name__ == "__main__":
    main()

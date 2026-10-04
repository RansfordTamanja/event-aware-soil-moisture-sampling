"""
prepare_data.py

Loads the three real sources, applies quality control, aligns them on the
native 15-min grid over the window covered by all three, and characterises
the data.

Outputs:
  outputs/data/aligned.csv          time, Port1..Port6, rain_mm (15 min), irr_lps, trigger
  outputs/data/data_summary.json    coverage, gaps, events, correlations
  outputs/tables/table_data.{csv,md}
"""

import json

import numpy as np
import pandas as pd

from config import (DATA, DEPTH_CM, INPUTS, PORTS, RAIN_TRIGGER_MM, RAIN_VALID_FLAGS, STEP_MIN, TABLES,
                    WET_REFRACTORY_STEPS, WET_RISE, WET_WINDOW_STEPS)


def load():
    s = pd.read_excel(INPUTS / "SoilWaterContent.xlsx")
    s["Timestamps"] = pd.to_datetime(s.Timestamps)
    s = s.set_index("Timestamps")[PORTS].sort_index()
    s = s[~s.index.duplicated()]

    r = pd.read_excel(INPUTS / "Gallipoli_P_5 min - T_15_min - VV_VD_15_min.xlsx", sheet_name="P_5min")
    r.columns = ["t", "p", "flag"]
    r["t"] = pd.to_datetime(r.t)
    n_flag_bad = int((~r.flag.isin(RAIN_VALID_FLAGS)).sum())
    r.loc[~r.flag.isin(RAIN_VALID_FLAGS), "p"] = np.nan
    # 5-min accumulations ending at t -> 15-min totals ending at t
    rain = r.set_index("t").p.resample(f"{STEP_MIN}min", label="right", closed="right").sum(min_count=1)

    q = pd.read_excel(INPUTS / "TREND_Portata_Uliveto_date_convertito.xlsx")
    q.columns = ["flow", "t"]
    q["t"] = pd.to_datetime(q.t)
    irr = q.set_index("t").flow.sort_index()
    irr = irr[~irr.index.duplicated()]
    return s, rain, irr, n_flag_bad


def wetting_onsets(x):
    """Indices where the series rises by more than WET_RISE within WET_WINDOW_STEPS."""
    x = np.asarray(x, float)
    n = len(x)
    rise = np.zeros(n, bool)
    rise[WET_WINDOW_STEPS:] = (x[WET_WINDOW_STEPS:] - x[:-WET_WINDOW_STEPS]) > WET_RISE
    on, last = [], -10 ** 9
    for i in np.flatnonzero(rise):
        if i - last > WET_REFRACTORY_STEPS:
            on.append(i - WET_WINDOW_STEPS)     # onset = start of the rising window
        last = i
    return np.array(on, int)


def main():
    s, rain, irr, n_flag_bad = load()
    t0 = max(s.index.min(), irr.index.min()).ceil(f"{STEP_MIN}min")
    t1 = min(s.index.max(), irr.index.max(), rain.index.max()).floor(f"{STEP_MIN}min")
    grid = pd.date_range(t0, t1, freq=f"{STEP_MIN}min")
    soil = s.reindex(grid)
    # The logger lost most of its data from November 2022 onwards (52-92 % of each
    # winter month missing). The study window ends on the last day before the first
    # day with more than 10 % of its native samples missing.
    daily_missing = soil[PORTS[0]].isna().resample("D").mean()
    bad_days = daily_missing[daily_missing > 0.10].index
    winter_missing = {str(k): float(v) for k, v in soil[PORTS[0]].isna().groupby(soil.index.month).mean().items()}
    if len(bad_days):
        t1 = bad_days[0] - pd.Timedelta(minutes=STEP_MIN)
        grid = pd.date_range(t0, t1, freq=f"{STEP_MIN}min")
        soil = soil.loc[grid]
    n_soil_gap = int(soil[PORTS[0]].isna().sum())
    soil = soil.interpolate(limit=4, limit_area="inside")      # fill logger dropouts up to 1 h
    df = soil.copy()
    df["rain_mm"] = rain.reindex(grid).fillna(0.0)
    df["irr_lps"] = irr.reindex(grid).ffill(limit=1).fillna(0.0)
    df["trigger"] = ((df.rain_mm >= RAIN_TRIGGER_MM) | (df.irr_lps > 0)).astype(int)
    df.index.name = "time"
    df.to_csv(DATA / "aligned.csv")

    rows = []
    for p in PORTS:
        x = df[p]
        on = wetting_onsets(x.values)
        rows.append({"port": p, "depth_cm": DEPTH_CM[p], "mean": x.mean(), "min": x.min(), "max": x.max(),
                     "lag1_autocorr": x.autocorr(1), "share_steps_unchanged": float((x.diff().abs() < 1e-9).mean()),
                     "wetting_onsets": len(on)})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table_data.csv", index=False)
    (TABLES / "table_data.md").write_text(t.to_markdown(index=False, floatfmt=".3f"))
    daily_rain = df.rain_mm.resample("D").sum()
    summary = {
        "start": str(grid[0]), "end": str(grid[-1]), "days": len(grid) / 96, "steps": len(grid),
        "soil_gap_steps_filled": n_soil_gap, "missing_share_by_month_full_record": winter_missing,
        "remaining_nan_steps": int(df[PORTS[0]].isna().sum()), "rain_flagged_invalid_5min": n_flag_bad,
        "rain_total_mm": float(df.rain_mm.sum()), "rain_days_ge_1mm": int((daily_rain >= 1).sum()),
        "irrigation_share_of_steps": float((df.irr_lps > 0).mean()),
        "irrigation_days": int((df.irr_lps.resample("D").max() > 0).sum()),
        "trigger_share_of_steps": float(df.trigger.mean()),
        "port_corr": df[PORTS].corr().round(3).to_dict(),
    }
    with open(DATA / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "port_corr"}, indent=1))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

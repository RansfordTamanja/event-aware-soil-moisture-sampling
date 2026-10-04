"""
exp_main.py

Experiment 1: every policy family is evaluated over its parameter grid on the
calibration weeks (every CAL_EVERY-th week, interleaved across the season) and
on the test weeks (all others). Policies run continuously over the whole
record; only the scoring is split.

Equal-rate comparison: for each target sampling rate R, each family's
configuration is chosen on the CALIBRATION period only (the configuration with
the lowest calibration RMSE among those whose calibration rate does not exceed
R) and is then scored on the TEST period. No choice uses test data.

Outputs:
  outputs/data/grid_results.json     every configuration, calibration and test
  outputs/data/selected.json         selected configurations per family and rate
  outputs/data/day_errors.json       per-day squared errors at the main rate
  outputs/tables/table_equal_rate.*  test results at each target rate
  outputs/tables/table_main.*        test results at the main rate (R = 0.10)
  outputs/tables/table_depth.*       main rate, by sensor depth
"""

import json

import numpy as np
import pandas as pd

from config import (CAL_EVERY, DATA, DELTA_SWEEP, DEPTH_CM, FIXED_INTERVALS, HOLD_MIN, HOLD_SWEEP, PORTS,
                    RAIN_THR_SWEEP, RAIN_TRIGGER_MM, TABLES, TARGET_RATES, TEVENT_SWEEP, TMAX_SWEEP)
from evaluate import DAY_STEPS, score
from policies import adaptive, event_aware, event_reactive, fixed, oracle_sod
from prepare_data import wetting_onsets

MAIN_RATE = 0.10
FAMILIES = ["Fixed", "Reactive", "Event-aware", "Event-aware + reactive"]


def load():
    d = pd.read_csv(DATA / "aligned.csv", parse_dates=["time"]).set_index("time")
    return d


def configs(family):
    if family == "Fixed":
        return [{"T": T} for T in FIXED_INTERVALS + [720, 960]]
    if family == "Reactive":
        return [{"delta": dl, "tmax": tm} for dl in DELTA_SWEEP for tm in TMAX_SWEEP]
    if family == "Event-aware":
        return [{"tmax": tm, "tevent": te, "hold": h, "rain_thr": r} for tm in TMAX_SWEEP
                for te in TEVENT_SWEEP for h in HOLD_SWEEP for r in RAIN_THR_SWEEP]
    if family == "Event-aware + reactive":
        return [{"delta": dl, "tmax": tm, "tevent": te, "hold": h, "rain_thr": RAIN_TRIGGER_MM}
                for dl in DELTA_SWEEP for tm in TMAX_SWEEP for te in TEVENT_SWEEP for h in (0, 60, 180)]
    if family == "Oracle SoD":
        return [{"delta": dl} for dl in [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.016]]
    raise ValueError(family)


def make_trigger(d, rain_thr=RAIN_TRIGGER_MM, latency_min=0, use_rain=True, use_irr=True):
    """Trigger series seen by the node. latency_min delays every trigger (the node
    learns of an event latency_min after it is recorded)."""
    g = np.zeros(len(d), bool)
    if use_rain:
        g |= d.rain_mm.values >= rain_thr
    if use_irr:
        g |= d.irr_lps.values > 0
    L = int(round(latency_min / 15))
    if L > 0:
        g = np.concatenate([np.zeros(L, bool), g[:-L]])
    return g


def run_policy(family, cfg, x, g, hold=HOLD_MIN):
    hold = cfg.get("hold", hold)
    if family == "Fixed":
        return fixed(x, cfg["T"])
    if family == "Reactive":
        return adaptive(x, cfg["delta"], tmax=cfg["tmax"])
    if family == "Event-aware":
        return event_aware(x, g, tmax=cfg["tmax"], hold=hold, tevent=cfg["tevent"])
    if family == "Event-aware + reactive":
        return event_reactive(x, g, cfg["delta"], tmax=cfg["tmax"], hold=hold, tevent=cfg["tevent"])
    if family == "Oracle SoD":
        return oracle_sod(x, cfg["delta"])


def aggregate(results):
    """Mean over ports for continuous metrics; pooled over events for event metrics."""
    lat = np.concatenate([r["latencies"] for r in results]) if any(r["latencies"] for r in results) else np.array([])
    pk = np.concatenate([r["peak_errs"] for r in results]) if any(r["peak_errs"] for r in results) else np.array([])
    ev = sum(r["events"] for r in results)
    det = sum(r["detected"] for r in results)
    return {
        "rate": float(np.mean([r["rate"] for r in results])),
        "rmse_interp": float(np.mean([r["rmse_interp"] for r in results])),
        "rmse_hold": float(np.mean([r["rmse_hold"] for r in results])),
        "maxerr_hold": float(np.mean([r["maxerr_hold"] for r in results])),
        "events": int(ev), "detected": int(det), "recall": det / ev if ev else np.nan,
        "delay_mean_min": float(lat.mean()) if lat.size else np.nan,
        "delay_p90_min": float(np.percentile(lat, 90)) if lat.size else np.nan,
        "peak_err_mean": float(pk.mean()) if pk.size else np.nan,
        "peak_err_p90": float(np.percentile(pk, 90)) if pk.size else np.nan,
    }


def split_masks(n):
    week = np.arange(n) // (7 * DAY_STEPS)
    cal = week % CAL_EVERY == 0
    return cal, ~cal


def schedules(d, family, cfg, g=None, hold=HOLD_MIN):
    if g is None:
        g = make_trigger(d, rain_thr=cfg.get("rain_thr", RAIN_TRIGGER_MM))
    return {p: run_policy(family, cfg, d[p].values, g, hold=hold) for p in PORTS}


def evaluate_schedules(d, S_by_port, masks, onsets, keep=False, logger=False):
    """Score precomputed schedules on several masks. logger=True: the six ports
    share one datalogger, which reads all ports at every wake-up, so every port
    is sampled at the union of the wake-up times."""
    if logger:
        U = np.unique(np.concatenate(list(S_by_port.values())))
        S_by_port = {p: U for p in PORTS}
    out = {}
    for name, m in masks.items():
        per = {p: score(d[p].values, S_by_port[p], m, onsets=onsets[p]) for p in PORTS}
        agg = aggregate(list(per.values()))
        if logger:
            agg["rate"] = float(m[S_by_port[PORTS[0]]].sum() / m.sum())
        if keep:
            agg["_per_port"] = per
        out[name] = agg
    return out


def evaluate_config(d, family, cfg, mask, onsets, hold=HOLD_MIN, keep=False, g=None, logger=False):
    S = schedules(d, family, cfg, g=g, hold=hold)
    return evaluate_schedules(d, S, {"m": mask}, onsets, keep=keep, logger=logger)["m"]


def select(grid, family, R):
    cands = [c for c in grid[family] if c["cal"]["rate"] <= R + 1e-9]
    if not cands:
        return None
    return min(cands, key=lambda c: c["cal"]["rmse_interp"])


def main():
    d = load()
    N = len(d)
    cal_m, test_m = split_masks(N)
    onsets = {p: wetting_onsets(d[p].values) for p in PORTS}
    grid = {}
    for fam in FAMILIES + ["Oracle SoD"]:
        grid[fam] = []
        for cfg in configs(fam):
            S = schedules(d, fam, cfg)
            port = evaluate_schedules(d, S, {"cal": cal_m, "test": test_m}, onsets)
            node = evaluate_schedules(d, S, {"cal": cal_m, "test": test_m}, onsets, logger=True)
            grid[fam].append({"cfg": cfg, "cal": port["cal"], "test": port["test"],
                              "logger_cal": node["cal"], "logger_test": node["test"]})
        print(f"{fam}: {len(grid[fam])} configurations")
    with open(DATA / "grid_results.json", "w") as f:
        json.dump(grid, f, indent=1, default=float)

    rows, selected = [], {}
    for R in TARGET_RATES:
        for fam in FAMILIES:
            c = select(grid, fam, R)
            if c is None:
                continue
            selected.setdefault(fam, {})[str(R)] = c["cfg"]
            t = c["test"]
            rows.append({"target_rate": R, "policy": fam, "config": json.dumps(c["cfg"]),
                         "cal_rate": c["cal"]["rate"], **{k: t[k] for k in
                         ["rate", "rmse_interp", "rmse_hold", "maxerr_hold", "recall",
                          "delay_mean_min", "delay_p90_min", "peak_err_mean", "peak_err_p90", "events"]}})
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "table_equal_rate.csv", index=False)
    (TABLES / "table_equal_rate.md").write_text(tab.to_markdown(index=False, floatfmt=".4f"))
    # logger level: the datalogger reads all six ports at each wake-up, so energy
    # follows the union of the ports' wake-up times; selection uses logger-level
    # calibration results only
    lrows = []
    for R in TARGET_RATES:
        for fam in FAMILIES:
            cands = [c for c in grid[fam] if c["logger_cal"]["rate"] <= R + 1e-9]
            if not cands:
                continue
            c = min(cands, key=lambda c: c["logger_cal"]["rmse_interp"])
            t = c["logger_test"]
            lrows.append({"target_rate": R, "policy": fam, "config": json.dumps(c["cfg"]),
                          "cal_rate": c["logger_cal"]["rate"], **{k: t[k] for k in
                          ["rate", "rmse_interp", "rmse_hold", "maxerr_hold", "recall", "delay_mean_min",
                           "delay_p90_min", "peak_err_mean"]}})
    lt = pd.DataFrame(lrows)
    lt.to_csv(TABLES / "table_equal_rate_logger.csv", index=False)
    (TABLES / "table_equal_rate_logger.md").write_text(lt.to_markdown(index=False, floatfmt=".4f"))
    print("\nLOGGER LEVEL\n", lt.round(4).to_string(index=False))

    main_tab = tab[tab.target_rate == MAIN_RATE].drop(columns="target_rate")
    main_tab.to_csv(TABLES / "table_main.csv", index=False)
    (TABLES / "table_main.md").write_text(main_tab.to_markdown(index=False, floatfmt=".4f"))
    with open(DATA / "selected.json", "w") as f:
        json.dump(selected, f, indent=1)
    print(tab.round(4).to_string(index=False))

    # equal-rate comparison on the test weeks: lower envelope of each family's
    # (rate, error) points, interpolated on log-log axes at common rates
    env_rows = []
    for R in [0.05, 0.075, 0.10, 0.15, 0.20, 0.25]:
        row = {"rate": R}
        for fam in FAMILIES + ["Oracle SoD"]:
            for metric, key in [("rmse_interp", "test"), ("rmse_hold", "test"),
                                ("logger_rmse_interp", "logger_test"), ("logger_rmse_hold", "logger_test")]:
                if fam == "Oracle SoD" and key == "logger_test":
                    continue
                m_ = metric.replace("logger_", "")
                pts = sorted((c[key]["rate"], c[key][m_]) for c in grid[fam])
                env, best = [], np.inf
                for r_, e_ in pts:
                    if e_ < best:
                        env.append((r_, e_)); best = e_
                rr = np.array([p[0] for p in env]); ee = np.array([p[1] for p in env])
                ok = ee > 0
                rr, ee = rr[ok], ee[ok]
                row[f"{metric}[{fam}]"] = (float(np.exp(np.interp(np.log(R), np.log(rr), np.log(ee))))
                                           if rr.size and rr.min() <= R <= rr.max() else np.nan)
        env_rows.append(row)
    et = pd.DataFrame(env_rows)
    et.to_csv(TABLES / "table_envelope.csv", index=False)
    (TABLES / "table_envelope.md").write_text(et.to_markdown(index=False, floatfmt=".5f"))
    print(et.round(5).to_string(index=False))

    # per-day errors and per-event delays at the main rate (for significance tests);
    # fixed 120-min sampling is added as a conservative baseline that uses MORE
    # samples than the event-aware configuration selected at the main rate
    selected.setdefault("Fixed-120", {})[str(MAIN_RATE)] = {"T": 120}
    day_err, events, depth_rows = {}, {}, []
    for fam in FAMILIES + ["Fixed-120"]:
        cfg = selected.get(fam, {}).get(str(MAIN_RATE))
        if cfg is None:
            continue
        agg = evaluate_config(d, "Fixed" if fam == "Fixed-120" else fam, cfg, test_m, onsets, keep=True)
        per = agg["_per_port"]
        sq = np.mean([per[p]["sq_err_interp"] for p in PORTS], axis=0)
        sqh = np.mean([per[p]["sq_err_hold"] for p in PORTS], axis=0)
        days = (np.arange(N) // DAY_STEPS)[test_m]
        day_err[fam] = {"interp": pd.Series(sq).groupby(days).mean().tolist(),
                        "hold": pd.Series(sqh).groupby(days).mean().tolist()}
        events[fam] = {p: per[p]["latencies"] for p in PORTS}
        for depth in (30, 60):
            ps = [p for p in PORTS if DEPTH_CM[p] == depth]
            a = aggregate([per[p] for p in ps])
            depth_rows.append({"policy": fam, "depth_cm": depth, **{k: a[k] for k in
                               ["rate", "rmse_interp", "rmse_hold", "recall", "delay_mean_min", "events"]}})
    with open(DATA / "day_errors.json", "w") as f:
        json.dump({"main_rate": MAIN_RATE, "day_err": day_err, "event_delays": events}, f, default=float)
    dt = pd.DataFrame(depth_rows)
    dt.to_csv(TABLES / "table_depth.csv", index=False)
    (TABLES / "table_depth.md").write_text(dt.to_markdown(index=False, floatfmt=".4f"))
    print(dt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

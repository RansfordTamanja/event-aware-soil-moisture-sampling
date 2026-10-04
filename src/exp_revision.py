"""
exp_revision.py

Analyses added in revision.

A  Event-count reconciliation: wetting onsets per port, in test weeks, more than
   one day before the end of the record, evaluable (the native series exceeds the
   onset value by kappa within 24 h), and detected by each policy.
B  Realised operating points at every budget: selected configuration with its
   calibration rate and error and its test rate and error, for every family.
   Strict-budget check: event-aware and hybrid configurations reselected with
   tighter calibration caps (0.090, 0.095) so that the test rate stays within the
   nominal 10 percent budget.
C  Cluster bootstrap (resampling whole test days, all six ports together, 2,000
   resamples) for the 10 percent comparison: interpolation and hold RMSE ratios,
   mean daily MSE difference, and mean and median detection-delay differences.
D  Event-definition sensitivity: onset rise {0.005, 0.01, 0.02} m3/m3 within 2 h,
   separation {2, 4, 8} h, detection threshold kappa {0.0025, 0.005, 0.01} m3/m3.
E  Leave-one-sensor-out: configurations selected on calibration weeks of five
   ports and scored on test weeks of the held-out port; depth stratification.
Outputs: outputs/tables/table_rev_*.csv / .md
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from config import *
from exp_main import load, configs, schedules, evaluate_schedules, split_masks, make_trigger, run_policy
from evaluate import score, reconstruct, DAY_STEPS
from prepare_data import wetting_onsets

RNG = np.random.default_rng(SEED)
FAMS = ["Fixed", "Reactive", "Event-aware", "Event-aware + reactive", "Oracle SoD"]


def save(df, name):
    df.to_csv(TABLES / f"table_rev_{name}.csv", index=False)
    (TABLES / f"table_rev_{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4g"))
    print(f"\n== {name}\n" + df.round(5).to_string(index=False), flush=True)


def onsets_param(x, rise, win, refr):
    x = np.asarray(x, float); n = len(x)
    r = np.zeros(n, bool); r[win:] = (x[win:] - x[:-win]) > rise
    on, last = [], -10 ** 9
    for i in np.flatnonzero(r):
        if i - last > refr:
            on.append(i - win)
        last = i
    return np.array(on, int)


def delays(x, S, ons, mask, kappa):
    """Extra delay (min) for each evaluable onset in mask; dict onset->delay (nan if missed)."""
    out = {}
    for o in ons:
        if not mask[o] or o >= len(x) - DAY_STEPS:
            continue
        nat = np.flatnonzero(x[o:o + DAY_STEPS] - x[o] >= kappa)
        if not nat.size:
            continue
        w = S[(S >= o) & (S < o + DAY_STEPS)]
        hit = w[x[w] - x[o] >= kappa]
        out[o] = (hit[0] - (o + nat[0])) * STEP_MIN if hit.size else np.nan
    return out


def main():
    d = load(); N = len(d)
    cal_m, test_m = split_masks(N)
    ons = {p: wetting_onsets(d[p].values) for p in PORTS}
    sel = json.load(open(DATA / "selected.json"))
    day = np.arange(N) // DAY_STEPS
    # ---------------- schedules at the 10 percent budget
    cfg10 = {f: sel[f]["0.1"] for f in FAMS if f in sel and sel[f].get("0.1")}
    S10 = {f: schedules(d, f, cfg10[f]) for f in cfg10}
    S10["Fixed 120"] = schedules(d, "Fixed", {"T": 120})
    # ---------------- A event reconciliation
    rows = []
    for p in PORTS:
        x = d[p].values; o = ons[p]
        in_test = o[test_m[o]]
        before_end = in_test[in_test < N - DAY_STEPS]
        ev = delays(x, S10["Event-aware"][p], o, test_m, WET_DETECT)
        rows.append({"port": p, "depth_cm": DEPTH_CM[p], "onsets_all": len(o), "in_test_weeks": len(in_test),
                     "before_last_day": len(before_end), "evaluable": len(ev),
                     **{f"detected_{k}": int(np.sum(~np.isnan(list(delays(x, S10[k][p], o, test_m, WET_DETECT).values()))))
                        for k in ["Fixed", "Reactive", "Event-aware", "Fixed 120"]}})
    A = pd.DataFrame(rows); A.loc[len(A)] = ["all", "", *A.iloc[:, 2:].sum().values]
    save(A, "events")
    # ---------------- B realised operating points
    grid = json.load(open(DATA / "grid_results.json"))
    rows = []
    for f in FAMS:
        for R in TARGET_RATES:
            c = sel.get(f, {}).get(str(R))
            if not c:
                continue
            S = schedules(d, f, c)
            ev = evaluate_schedules(d, S, {"cal": cal_m, "test": test_m}, ons)
            lat = np.concatenate([[v for v in delays(d[p].values, S[p], ons[p], test_m, WET_DETECT).values() if not np.isnan(v)] for p in PORTS])
            rows.append({"family": f, "budget": R, "config": json.dumps(c), "cal_rate": ev["cal"]["rate"],
                         "cal_rmse_interp": ev["cal"]["rmse_interp"], "test_rate": ev["test"]["rate"],
                         "test_rmse_interp": ev["test"]["rmse_interp"], "test_rmse_hold": ev["test"]["rmse_hold"],
                         "delay_mean": lat.mean(), "delay_median": np.median(lat),
                         "delay_q1": np.percentile(lat, 25), "delay_q3": np.percentile(lat, 75)})
    save(pd.DataFrame(rows), "operating_points")
    # strict budget: tighter calibration caps
    rows = []
    for f in ["Event-aware", "Event-aware + reactive"]:
        cands = grid[f]
        for cap in [0.10, 0.095, 0.09, 0.085, 0.08]:
            ok = [c for c in cands if c["cal"]["rate"] <= cap + 1e-9]
            if not ok:
                rows.append({"family": f, "cal_cap": cap, "config": "none within cap"}); continue
            best = min(ok, key=lambda c: c["cal"]["rmse_interp"])
            cfg = best["cfg"]
            S = schedules(d, f, cfg)
            ev = evaluate_schedules(d, S, {"cal": cal_m, "test": test_m}, ons)["test"]
            rows.append({"family": f, "cal_cap": cap, "config": json.dumps(cfg), "cal_rate": best["cal"]["rate"],
                         "test_rate": ev["rate"], "test_rmse_interp": ev["rmse_interp"], "test_rmse_hold": ev["rmse_hold"],
                         "delay_mean": ev["delay_mean_min"]})
    save(pd.DataFrame(rows), "strict_budget")
    # ---------------- C cluster bootstrap over test days
    tdays = np.unique(day[test_m])
    def sq(S, p):
        interp, hold = reconstruct(d[p].values, S)
        x = d[p].values
        return (interp - x) ** 2, (hold - x) ** 2
    E = {k: {p: sq(S10[k][p], p) for p in PORTS} for k in ["Event-aware", "Fixed", "Fixed 120", "Reactive"]}
    idx_by_day = {t: np.flatnonzero((day == t) & test_m) for t in tdays}
    lat = {k: {p: delays(d[p].values, S10[k][p], ons[p], test_m, WET_DETECT) for p in PORTS} for k in E}
    def stats(days):
        ix = np.concatenate([idx_by_day[t] for t in days])
        out = {}
        for k in ["Fixed", "Fixed 120", "Reactive"]:
            ri = np.mean([np.sqrt(E[k][p][0][ix].mean()) for p in PORTS]) / np.mean([np.sqrt(E["Event-aware"][p][0][ix].mean()) for p in PORTS])
            rh = np.mean([np.sqrt(E[k][p][1][ix].mean()) for p in PORTS]) / np.mean([np.sqrt(E["Event-aware"][p][1][ix].mean()) for p in PORTS])
            # daily MSE (hold) difference, averaged over ports and days
            dm = np.mean([np.mean([E[k][p][1][idx_by_day[t]].mean() - E["Event-aware"][p][1][idx_by_day[t]].mean() for p in PORTS]) for t in days])
            a, b = [], []
            for p in PORTS:
                for o, v in lat["Event-aware"][p].items():
                    if day[o] in set(days) and o in lat[k][p] and not np.isnan(v) and not np.isnan(lat[k][p][o]):
                        a.append(lat[k][p][o] * list(days).count(day[o])); b.append(v * list(days).count(day[o]))
            # weighted by resample multiplicity: expand
            aa, bb = [], []
            cnt = pd.Series(days).value_counts()
            for p in PORTS:
                for o, v in lat["Event-aware"][p].items():
                    m = cnt.get(day[o], 0)
                    if m and o in lat[k][p] and not np.isnan(v) and not np.isnan(lat[k][p][o]):
                        aa += [lat[k][p][o]] * m; bb += [v] * m
            aa, bb = np.array(aa), np.array(bb)
            out[k] = (ri, rh, dm, (aa - bb).mean() if aa.size else np.nan, np.median(aa) - np.median(bb) if aa.size else np.nan)
        return out
    point = stats(list(tdays))
    boots = [stats(list(RNG.choice(tdays, len(tdays)))) for _ in range(2000)]
    rows = []
    names = ["rmse_interp_ratio", "rmse_hold_ratio", "daily_mse_hold_diff", "delay_mean_diff_min", "delay_median_diff_min"]
    for k in point:
        for j, nm in enumerate(names):
            v = np.array([b[k][j] for b in boots]); v = v[~np.isnan(v)]
            rows.append({"comparison": f"{k} vs Event-aware", "quantity": nm, "estimate": point[k][j],
                         "ci_lo": np.percentile(v, 2.5), "ci_hi": np.percentile(v, 97.5)})
    save(pd.DataFrame(rows), "bootstrap")
    # ---------------- D event-definition sensitivity
    rows = []
    for rise in [0.005, 0.01, 0.02]:
        for refr_h in [2, 4, 8]:
            for kap in [0.0025, 0.005, 0.01]:
                r = {"rise": rise, "separation_h": refr_h, "kappa": kap}
                for k in ["Event-aware", "Fixed", "Fixed 120", "Reactive"]:
                    L, n = [], 0
                    for p in PORTS:
                        o = onsets_param(d[p].values, rise, WET_WINDOW_STEPS, refr_h * 4)
                        dd = delays(d[p].values, S10[k][p], o, test_m, kap)
                        n += len(dd); L += [v for v in dd.values() if not np.isnan(v)]
                    r["events"] = n; r[f"{k}_mean"] = np.mean(L) if L else np.nan; r[f"{k}_median"] = np.median(L) if L else np.nan
                    r[f"{k}_detected"] = len(L)
                rows.append(r)
    save(pd.DataFrame(rows), "event_definition")
    # ---------------- E leave-one-sensor-out
    per = {}
    for f in ["Fixed", "Reactive", "Event-aware"]:
        per[f] = []
        for c in configs(f):
            S = schedules(d, f, c)
            rec = {"cfg": c}
            for p in PORTS:
                rec[p] = {m: score(d[p].values, S[p], mk, onsets=ons[p]) for m, mk in [("cal", cal_m), ("test", test_m)]}
                for m in ("cal", "test"):
                    rec[p][m] = {kk: rec[p][m][kk] for kk in ("rate", "rmse_interp", "rmse_hold")}
            per[f].append(rec)
    rows = []
    for hold_out in PORTS:
        others = [p for p in PORTS if p != hold_out]
        r = {"held_out": hold_out, "depth_cm": DEPTH_CM[hold_out]}
        for f in per:
            ok = [c for c in per[f] if np.mean([c[p]["cal"]["rate"] for p in others]) <= 0.10 + 1e-9]
            best = min(ok, key=lambda c: np.mean([c[p]["cal"]["rmse_interp"] for p in others]))
            r[f"{f}_rate"] = best[hold_out]["test"]["rate"]; r[f"{f}_rmse_interp"] = best[hold_out]["test"]["rmse_interp"]
            r[f"{f}_rmse_hold"] = best[hold_out]["test"]["rmse_hold"]
        r["ratio_fixed_event_interp"] = r["Fixed_rmse_interp"] / r["Event-aware_rmse_interp"]
        r["ratio_reactive_event_interp"] = r["Reactive_rmse_interp"] / r["Event-aware_rmse_interp"]
        rows.append(r)
    save(pd.DataFrame(rows), "loso")


if __name__ == "__main__":
    main()

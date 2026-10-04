"""
significance_testing.py

Tests at the main rate (R = 0.10), on the test weeks.

Day-level: paired day-block bootstrap of the difference in daily mean squared
error (5,000 resamples) and an exact two-sided sign test on days.
Event-level: paired Wilcoxon signed-rank test of extra detection delays over
the wetting events (same events for every policy).

Comparisons: Event-aware vs Fixed-120 (a baseline with MORE samples),
Event-aware vs Reactive, Event-aware vs Fixed (rate-matched selection).

Outputs: outputs/data/significance.json, outputs/tables/table_significance.*
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import binom, wilcoxon

from config import DATA, N_BOOT, PORTS, SEED, TABLES


def paired(a, b, rng):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    boot = diff[idx].mean(1)
    p = max(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 2 / N_BOOT)
    nz = diff[diff != 0]
    k = int((nz < 0).sum())
    sign_p = float(min(1, 2 * min(binom.cdf(k, len(nz), .5), binom.sf(k - 1, len(nz), .5)))) if len(nz) else np.nan
    return {"days": len(diff), "mean_diff": float(diff.mean()), "ratio_of_means": float(a.mean() / b.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "p_boot": float(min(p, 1)), "A_better_days": k, "non_tied": int(len(nz)), "sign_p": sign_p}


def main():
    rng = np.random.default_rng(SEED)
    J = json.load(open(DATA / "day_errors.json"))
    de, ev = J["day_err"], J["event_delays"]
    rows = []
    for other in ["Fixed-120", "Reactive", "Fixed"]:
        if other not in de:
            continue
        for metric in ["interp", "hold"]:
            r = paired(de["Event-aware"][metric], de[other][metric], rng)
            rows.append({"comparison": f"Event-aware vs {other}", "metric": f"daily MSE ({metric})", **r})
        a = np.concatenate([ev["Event-aware"][p] for p in PORTS])
        b = np.concatenate([ev[other][p] for p in PORTS])
        w = wilcoxon(a, b, zero_method="wilcox") if np.any(a != b) else None
        rows.append({"comparison": f"Event-aware vs {other}", "metric": "event detection delay (min)",
                     "days": len(a), "mean_diff": float((a - b).mean()),
                     "ratio_of_means": float(a.mean() / b.mean()) if b.mean() > 0 else np.nan,
                     "A_better_days": int((a < b).sum()), "non_tied": int((a != b).sum()),
                     "p_boot": np.nan, "sign_p": np.nan,
                     "ci_lo": np.nan, "ci_hi": np.nan,
                     "wilcoxon_p": float(w.pvalue) if w else np.nan})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))

    # where the gains arise: event-affected days (trigger on the day or the day
    # before, in the same 24-h blocks used for the daily errors) vs quiet days
    from exp_main import load, split_masks
    from evaluate import DAY_STEPS
    d = load()
    _, test_m = split_masks(len(d))
    block = np.arange(len(d)) // DAY_STEPS
    trig = pd.Series(d.trigger.values).groupby(block).max()
    tb = np.unique(block[test_m])
    ev_day = (trig.reindex(tb).values > 0) | (trig.reindex(tb - 1).fillna(0).values > 0)
    brows = []
    for other in ["Fixed-120", "Reactive", "Fixed"]:
        for metric in ["interp", "hold"]:
            a = np.array(de["Event-aware"][metric]); b = np.array(de[other][metric])
            for lab, m in [("event-affected days", ev_day), ("quiet days", ~ev_day)]:
                brows.append({"comparison": f"Event-aware vs {other}", "metric": metric, "days": lab,
                              "n_days": int(m.sum()), "rmse_event_aware": float(np.sqrt(a[m].mean())),
                              "rmse_other": float(np.sqrt(b[m].mean())), "mse_ratio": float(a[m].mean() / b[m].mean()),
                              "event_aware_better_days": int((a[m] < b[m]).sum()),
                              "share_of_total_reduction": float((b - a)[m].sum() / (b - a).sum())})
    bt = pd.DataFrame(brows)
    print(bt.round(5).to_string(index=False))
    bt.to_csv(TABLES / "table_event_vs_quiet.csv", index=False)
    (TABLES / "table_event_vs_quiet.md").write_text(bt.to_markdown(index=False, floatfmt=".5f"))
    t.to_csv(TABLES / "table_significance.csv", index=False)
    (TABLES / "table_significance.md").write_text(t.to_markdown(index=False, floatfmt=".4g"))
    with open(DATA / "significance.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)


if __name__ == "__main__":
    main()

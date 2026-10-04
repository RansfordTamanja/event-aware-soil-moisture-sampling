"""
evaluate.py

Scores a set of sample indices against the native 15-min series.

Reconstruction at the server:
  interpolation  linear interpolation between samples (archival accuracy)
  hold           last sampled value carried forward (what a real-time
                 irrigation controller sees)

Metrics (computed over the native indices where the boolean mask is True):
  rate          samples / native samples
  energy_rel    phi + (1 - phi) * rate  (daily energy relative to 15-min sampling)
  rmse_interp, rmse_hold, maxerr_hold   in m3/m3
  events        true wetting onsets in the range
  recall        share of onsets detected within 24 h (a sample exceeds the
                value at onset by WET_DETECT)
  delay_min     extra detection delay: minutes between the first native
                (15-min) sample that shows the event and the first policy
                sample that shows it (0 = detected as early as possible)
  peak_err      underestimation of the event peak (true maximum minus the
                maximum sampled value within 24 h of onset), m3/m3
"""

import numpy as np

from config import PHI_DEFAULT, STEP_MIN, WET_DETECT
from prepare_data import wetting_onsets

DAY_STEPS = 24 * 60 // STEP_MIN


def reconstruct(x, S):
    n = len(x)
    idx = np.arange(n)
    interp = np.interp(idx, S, x[S])
    pos = np.searchsorted(S, idx, side="right") - 1
    hold = x[S][pos]
    return interp, hold


def score(x, S, mask, phi=PHI_DEFAULT, onsets=None):
    x = np.asarray(x, float)
    S = np.asarray(S, int)
    mask = np.asarray(mask, bool)
    interp, hold = reconstruct(x, S)
    e_i, e_h = (interp - x)[mask], (hold - x)[mask]
    rate = mask[S].sum() / mask.sum()
    if onsets is None:
        onsets = wetting_onsets(x)
    ons = onsets[mask[onsets] & (onsets < len(x) - DAY_STEPS)]
    lat, peak, det = [], [], 0
    for o in ons:
        base = x[o]
        w = S[(S >= o) & (S < o + DAY_STEPS)]
        hit = w[x[w] - base >= WET_DETECT]
        native = np.flatnonzero(x[o:o + DAY_STEPS] - base >= WET_DETECT)
        if hit.size and native.size:
            det += 1
            lat.append((hit[0] - (o + native[0])) * STEP_MIN)
        true_peak = x[o:o + DAY_STEPS].max()
        samp_peak = x[w].max() if w.size else base
        peak.append(true_peak - samp_peak)
    return {
        "rate": rate, "energy_rel": phi + (1 - phi) * rate,
        "rmse_interp": float(np.sqrt(np.mean(e_i ** 2))), "rmse_hold": float(np.sqrt(np.mean(e_h ** 2))),
        "maxerr_hold": float(np.max(np.abs(e_h))),
        "events": int(len(ons)), "detected": int(det),
        "recall": det / len(ons) if len(ons) else np.nan,
        "delay_mean_min": float(np.mean(lat)) if lat else np.nan,
        "delay_p90_min": float(np.percentile(lat, 90)) if lat else np.nan,
        "latencies": lat, "peak_errs": peak,
        "peak_err_mean": float(np.mean(peak)) if peak else np.nan,
        "sq_err_interp": e_i ** 2, "sq_err_hold": e_h ** 2, "mask": mask,
    }

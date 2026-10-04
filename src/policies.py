"""
policies.py

Sampling policies. Each takes one sensor's native 15-min series x (length N)
and, where needed, the exogenous trigger series g (1 while rain >= threshold
fell in the last 15 min or the irrigation valve was open), and returns the
sorted native indices at which the node wakes and samples. Index 0 is always
sampled.

  fixed(T)            one sample every T minutes
  adaptive(delta)     REACTIVE exponential back-off: after each sample the
                      interval doubles (up to T_MAX) if the value changed by at
                      most delta since the previous sample, and resets to T_MIN
                      otherwise
  event_aware         PROPOSED: the quiet interval T_MAX while no event is
                      active; the event interval (default T_MIN) while a
                      trigger is active and for HOLD minutes after it ends; a
                      trigger arriving during sleep wakes the node at once
  event_reactive(delta)  ABLATION (hybrid): adaptive(delta) plus event pre-emption: a
                      trigger wakes the node immediately, and the interval is
                      held at the event interval (default T_MIN) during the
                      trigger and for HOLD minutes after it, after which
                      back-off resumes from T_MIN
  oracle_sod(delta)   send-on-delta with continuous knowledge of the signal:
                      samples exactly when the value departs from the last
                      sample by more than delta. Not realisable by a sleeping
                      node; used only as a reference for the number of samples
                      needed to track the signal to within delta.
"""

import numpy as np

from config import HOLD_MIN, STEP_MIN, T_MAX, T_MIN


def _steps(minutes):
    return max(1, int(round(minutes / STEP_MIN)))


def fixed(x, T, **_):
    return np.arange(0, len(x), _steps(T))


def adaptive(x, delta, tmin=T_MIN, tmax=T_MAX, **_):
    n, i = len(x), 0
    I, prev = _steps(tmin), None
    out = []
    while i < n:
        out.append(i)
        if prev is not None:
            I = _steps(tmin) if abs(x[i] - prev) > delta else min(2 * I, _steps(tmax))
        prev = x[i]
        i += I
    return np.array(out)


def _active(g, hold):
    """Boolean array: trigger active now or within `hold` minutes after it ended."""
    g = np.asarray(g, bool)
    h = _steps(hold) if hold > 0 else 0
    act = g.copy()
    if h:
        last = -10 ** 9
        for j in range(len(g)):
            if g[j]:
                last = j
            elif j - last <= h:
                act[j] = True
    return act


def event_aware(x, g, tmin=T_MIN, tmax=T_MAX, hold=HOLD_MIN, tevent=None, **_):
    act = _active(g, hold)
    te = _steps(tevent if tevent else tmin)
    n, i, out = len(x), 0, []
    while i < n:
        out.append(i)
        step = te if act[i] else _steps(tmax)
        nxt = i + step
        if not act[i]:   # an event starting before the next scheduled wake pre-empts the sleep
            upcoming = np.flatnonzero(act[i + 1:nxt])
            if upcoming.size:
                nxt = i + 1 + upcoming[0]
        i = nxt
    return np.array(out)


def event_reactive(x, g, delta, tmin=T_MIN, tmax=T_MAX, hold=HOLD_MIN, tevent=None, **_):
    act = _active(g, hold)
    te = _steps(tevent if tevent else tmin)
    n, i = len(x), 0
    I, prev = _steps(tmin), None
    out = []
    while i < n:
        out.append(i)
        if act[i]:
            I = te
        elif prev is not None:
            I = _steps(tmin) if abs(x[i] - prev) > delta else min(2 * I, _steps(tmax))
        prev = x[i]
        nxt = i + I
        upcoming = np.flatnonzero(act[i + 1:nxt])
        if upcoming.size:                     # event pre-emption: wake at the trigger
            nxt = i + 1 + upcoming[0]
        i = nxt
    return np.array(out)


def oracle_sod(x, delta, **_):
    out, last = [0], x[0]
    for j in range(1, len(x)):
        if abs(x[j] - last) > delta:
            out.append(j)
            last = x[j]
    return np.array(out)

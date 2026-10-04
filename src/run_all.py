"""
run_all.py

Runs the complete pipeline of the article "Event-Aware Adaptive Sampling for Soil Moisture Wireless
Sensor Networks: Accuracy at Equal Sampling Budgets and the Energy Cost of Trigger Delivery", in order:

    prepare_data.py          QC, alignment on the 15-min grid, study window, data summary
    exp_main.py              policy grids, calibration-selected comparison, equal-rate envelopes,
                             per-day and per-event results
    exp_sensitivity.py       trigger sources, hold period, normalised energy overhead (break-even rates)
    exp_trigger.py           trigger latency, missed triggers (20 draws), trigger-reception overhead
    exp_seasons.py           out-of-period validation: summer-autumn 2021, winter 2021-22
    significance_testing.py  day-block bootstrap, sign tests, Wilcoxon, event vs quiet days
    exp_revision.py          event-count reconciliation, realised operating points at every budget,
                             strict-budget reselection, day-clustered bootstrap intervals,
                             27 wetting-event definitions, leave-one-sensor-out selection
    energy_hw.py             datasheet-grounded node energy model (SX1276, STM32L072, TEROS 10),
                             LoRaWAN Class B trigger-reception cost
    exp_rev2.py              missed-trigger bands (100 draws per loss probability), trigger energy budget
    plotting.py              original figure set
    figs_rev.py              article figures: larger labels, architecture, trigger budget, missed-trigger bands
Usage:  python run_all.py
"""
import subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["prepare_data.py", "exp_main.py", "exp_sensitivity.py", "exp_trigger.py", "exp_seasons.py",
         "significance_testing.py", "exp_revision.py", "energy_hw.py", "exp_rev2.py", "plotting.py", "figs_rev.py"]
t0 = time.time()
for s in STEPS:
    print(f"\n{'=' * 60}\nRunning {s}\n{'=' * 60}", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, "-W", "ignore", s], cwd=SRC).returncode != 0:
        sys.exit(f"FAILED at {s}")
    print(f"[{s} finished in {time.time() - t:.0f} s]", flush=True)
print(f"\nPipeline complete in {(time.time() - t0) / 60:.1f} min. See outputs/data, outputs/tables, outputs/figures.")

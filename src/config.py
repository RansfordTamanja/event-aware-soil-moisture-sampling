"""
config.py

Paths, constants and every modelling choice of the pipeline, stated once.

Paper: "Event-Aware Adaptive Sampling for Energy-Efficient Soil Moisture
Wireless Sensor Networks".

Data (Bruni, Selicato, Vivaldi, Vurro, Berardi, Zenodo, doi:10.5281/zenodo.20665025):
  SoilWaterContent.xlsx     volumetric water content (m3/m3) every 15 min from
                            six sensor ports of a datalogger in an experimental
                            field at Gallipoli (Italy); odd ports at 30 cm,
                            even ports at 60 cm depth
  Gallipoli_P_5 min ...     regional weather-station rainfall, 5 min, with
                            automatic validation flags
  TREND_Portata_Uliveto ... irrigation flow of the olive grove (l/s), 15 min
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUT = ROOT / "outputs"
DATA, FIGS, TABLES = OUT / "data", OUT / "figures", OUT / "tables"
for _d in (DATA, FIGS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

PORTS = ["Port1", "Port2", "Port3", "Port4", "Port5", "Port6"]
DEPTH_CM = {"Port1": 30, "Port2": 60, "Port3": 30, "Port4": 60, "Port5": 30, "Port6": 60}
STEP_MIN = 15                     # native sampling interval (ground truth)
RAIN_VALID_FLAGS = {1, 11}        # 1 = automatic validation: correct; 11 = manually validated correct

# Study window: the period covered by soil moisture, rainfall AND irrigation records
# Calibration uses every CAL_EVERY-th week of the study window (interleaved
# blocks), so that it spans spring, the summer irrigation season and the
# autumn rains; the remaining weeks form the test set. Policies always run
# continuously over the whole record; only the scoring is split.
CAL_EVERY = 4
# ---------------------------------------------------------------------------
# Sampling policies (all intervals are multiples of the native 15 min)
# ---------------------------------------------------------------------------
T_MIN = 15                        # shortest interval (min)
T_MAX = 480                       # longest interval in quiet periods (8 h)
FIXED_INTERVALS = [15, 30, 60, 120, 240, 480]
# Change thresholds in m3/m3. The sensors report in steps of 0.001, so 0 means
# "react to any change of one resolution step".
DELTA_SWEEP = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016]
TMAX_SWEEP = [120, 240, 480, 960]      # longest quiet-period interval (min)
TEVENT_SWEEP = [15, 30, 60]            # sampling interval while an event is active (min)
TARGET_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]   # equal-rate comparison points
RAIN_TRIGGER_MM = 0.2             # default rainfall in the last 15 min that counts as an event
HOLD_MIN = 180                    # default hold after a trigger ends (min)
HOLD_SWEEP = [0, 60, 180, 360]    # hold periods in the event-aware grid (selected on calibration weeks)
RAIN_THR_SWEEP = [0.2, 1.0]       # rain thresholds (mm per 15 min) in the event-aware grid
# Trigger timing. The main experiments assume an IDEAL trigger: the node learns of
# rain or irrigation in the same 15-min step in which it is recorded (zero latency,
# no missed triggers). exp_trigger.py relaxes both assumptions.
LATENCY_SWEEP = [0, 15, 30, 60, 120]      # trigger latency (min)
MISS_SWEEP = [0.0, 0.1, 0.25, 0.5]        # probability that a whole trigger episode is missed
N_MISS_SEEDS = 20
PSI_SWEEP = [0.0, 0.01, 0.02, 0.05, 0.1]  # trigger-reception overhead (share of native daily energy)

# ---------------------------------------------------------------------------
# Wetting events (ground truth on the native series)
# ---------------------------------------------------------------------------
WET_RISE = 0.01                   # m3/m3 rise ...
WET_WINDOW_STEPS = 8              # ... within 2 h marks a wetting onset
WET_DETECT = 0.005                # a sample shows the event once it exceeds the pre-event value by this
WET_REFRACTORY_STEPS = 16         # 4 h between distinct onsets

# ---------------------------------------------------------------------------
# Energy (normalised): daily energy relative to native 15-min sampling
#   E_rel = phi + (1 - phi) * samples / native_samples
# phi = share of the native schedule's energy spent asleep (not per-sample)
# ---------------------------------------------------------------------------
PHI_DEFAULT = 0.1
PHI_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

N_BOOT = 5000
SEED = 2026

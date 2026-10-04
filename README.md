# Event-Aware Adaptive Sampling for Soil Moisture Wireless Sensor Networks: Accuracy at Equal Sampling Budgets and the Energy Cost of Trigger Delivery

Reproducible pipeline for the article by R. O. Tamanja and K. S. Adu-Manu (Journal of Sensors).

Sampling policies for a sleeping soil moisture node (fixed, reactive, event-aware, event-aware + reactive,
and a send-on-delta reference) are replayed against real 15-minute records from an experimental olive grove
at Gallipoli, southern Italy, with station rainfall and the grove's irrigation flow, and coupled to a
datasheet-grounded energy model of a LoRaWAN node.

## Data

Bruni, Selicato, Vivaldi, Vurro, Berardi, "Time-series analysis and learning approaches for soil moisture
prediction in precision irrigation: Dataset and code", Zenodo, version 2, doi:10.5281/zenodo.20665025.
Download `data.zip` from https://zenodo.org/records/20665025 (folder `dati`) and place these files in `inputs/`:

| File | Content |
|---|---|
| `SoilWaterContent.xlsx` | volumetric water content, six ports (odd 30 cm, even 60 cm), every 15 min |
| `Gallipoli_P_5 min - T_15_min - VV_VD_15_min.xlsx` | station rainfall every 5 min with validation flags (sheet `P_5min`) |
| `TREND_Portata_Uliveto_date_convertito.xlsx` | irrigation flow of the olive grove, every 15 min, from 2022-04-07 |

## Run

```bash
python -m pip install -r requirements.txt     # Python 3.12; versions pinned in requirements.txt
cd src
python run_all.py
```

The full pipeline runs end to end without manual steps and writes to `outputs/data`, `outputs/tables`,
and `outputs/figures`.

## Pipeline and article mapping

| Script | Purpose | Article element |
|---|---|---|
| `config.py` | paths, study window, event definition, grids, seeds | Sections 4.1 to 4.3 |
| `prepare_data.py` | quality control, 15-min alignment, wetting onsets | Section 4.1 (data for Fig. 2) |
| `policies.py` | fixed, reactive, event-aware, hybrid, send-on-delta schedules | Section 3.3, Eqs. (4) to (8) |
| `evaluate.py` | rate, interpolation and held-value error, detection delay | Eqs. (1), (2); Section 3.4 |
| `exp_main.py` | grids, calibration-week selection, equal-rate envelopes | Fig. 3, Table 2, Appendix Table A1 |
| `exp_sensitivity.py` | trigger sources, hold period, fixed rate of equal error | Section 5.5, Fig. 6 |
| `exp_trigger.py` | trigger latency, episode definition for missed triggers | Section 5.6 |
| `exp_seasons.py` | out-of-period validation (winter 2021-22, summer-autumn 2021) | Section 5.8 |
| `significance_testing.py` | event vs quiet days, Wilcoxon signed-rank tests | Section 5.3, Section 5.4, Fig. 5 |
| `exp_revision.py` | event counts, operating points, strict budget, day-clustered bootstrap, 27 event definitions, leave-one-sensor-out | Tables 2 to 4, A1; Sections 5.3, 5.5 |
| `energy_hw.py` | datasheet energy model, LoRaWAN Class B reception cost | Section 3.5, Eqs. (10), (11); Section 5.7 |
| `exp_rev2.py` | missed-trigger bands (100 draws), trigger energy budget | Eq. (12); Figs. 7, 8; Section 5.7 |
| `plotting.py`, `figs_rev.py` | all figures (PDF and PNG) | Figs. 1 to 8 |
| `run_all.py` | runs everything in order | |

## Key settings

- Study window: 2022-04-07 to 2022-11-10 (217.6 days, 20,886 steps of 15 min, no missing soil samples).
- Calibration: every fourth week from 2022-04-07; test: all other weeks. Policies run on the whole record,
  scoring is split.
- Wetting onset: rise above 0.01 m3/m3 within 2 h, onsets less than 4 h apart merged; detection when the
  water content exceeds its onset value by 0.005 m3/m3. Alternatives in `exp_revision.py`.
- Seeds: 2026 for the cluster bootstrap and grid selection; 2033 for the 100-draw missed-trigger simulation.
- Energy model inputs (datasheets): SX1276 sleep 0.2 uA, RX 10.3 mA, TX 28 mA (+13/+14 dBm class);
  STM32L072 stop + RTC 0.86 uA, run 93 uA/MHz at 32 MHz; TEROS 10 12 mA for 10 ms (50 ms conservative);
  LoRaWAN Class B beacon period 128 s, ping slots of 8 symbols at SF9/125 kHz; supply 3.3 V.
  These are modelled estimates for a representative node, not measurements of the logged hardware.

## Main results (10 percent budget, test weeks)

- Event-aware sampling: test rate 0.106, interpolation RMSE 0.00055 m3/m3; fixed 4 h 0.00336, fixed 2 h 0.00198.
- Median extra detection delay 15 min (IQR 0 to 15) against 120 min (fixed 4 h) and 60 min (fixed 2 h).
- Interpolation RMSE ratio, fixed 2 h over event-aware: 3.57 (day-clustered 95 percent interval 2.43 to 4.20).
- Strict budget (calibration cap 0.095): test rate 0.099, interpolation RMSE 0.00061.
- Trigger energy budget: 0.71 to 0.89 uA (SF7), 2.27 to 2.45 uA (SF9), 16.2 to 16.3 uA (SF12);
  LoRaWAN Class B with 128 s pings draws 21.5 uA.

"""
energy_hw.py

Datasheet-grounded energy model for a representative LoRaWAN soil-moisture node
(modelled estimates, not hardware measurements).

Components and sources
  Radio  Semtech SX1276 (datasheet): sleep 0.2 uA; RX (LoRa, 125 kHz) 10.3 mA;
         TX 28 mA at +13 dBm (RFO), 87 mA at +17 dBm, 120 mA at +20 dBm; 3.3 V.
  MCU    STMicroelectronics STM32L072 (datasheet): Stop mode + RTC + RAM 0.86 uA;
         Run mode 93 uA/MHz (32 MHz -> 2.98 mA).
  Probe  METER TEROS 10 (manual): 12 mA at 3 V during excitation; measurement
         duration 10 ms (current manual), 25-50 ms (earlier manual).
  LoRaWAN Class B (specification, Semtech note): beacon every 128 s; ping slots of
         30 ms; a ping slot opens the receiver for 8 symbols to detect a preamble;
         default ping-slot data rate SF9/125 kHz.
Airtime: Semtech AN1200.13 formula (explicit header, CRC on, coding rate 4/5,
preamble 8 symbols, low-data-rate optimisation for SF11/SF12 at 125 kHz).

Assumptions (stated, varied in sensitivity)
  MCU active time per sample: 20 ms (wake, ADC, packet assembly), plus the probe time.
  One LoRaWAN uplink per sample, 11-byte application payload (13-byte MAC overhead),
  TX power +14 dBm class (28 mA); Class A receive windows: two windows of 8 symbols
  at the uplink data rate (no downlink).
  Board-level sleep floor: chip-level 1.06 uA (MCU + radio), and 10 uA and 50 uA
  to represent regulator and peripheral leakage of real boards.
  Battery: 2 x AA lithium, 3000 mAh usable at 3.3 V (self-discharge ignored).
Output: tables/table_energy_hw.csv/.md and table_energy_psi.csv/.md
"""
import math
import numpy as np, pandas as pd
from config import TABLES

V = 3.3
I_TX, I_RX, I_MCU, I_PROBE = 28e-3, 10.3e-3, 2.98e-3, 12e-3
I_SLEEP_CHIP = 0.86e-6 + 0.2e-6
CAP_AH = 3.0
NATIVE_PER_DAY = 96


def airtime(sf, payload_app=11, bw=125e3, cr=1, preamble=8):
    pl = payload_app + 13
    tsym = 2 ** sf / bw
    de = 1 if (sf >= 11 and bw == 125e3) else 0
    n = 8 + max(math.ceil((8 * pl - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * tsym + n * tsym


def charge_per_sample(sf, t_probe, t_mcu=0.020):
    """Coulombs per sample: probe + MCU + uplink + two Class A windows."""
    tsym = 2 ** sf / 125e3
    q = I_PROBE * t_probe + I_MCU * (t_mcu + t_probe)
    q += (I_TX + I_MCU) * airtime(sf)
    q += 2 * (I_RX + I_MCU) * 8 * tsym
    return q


def classb_charge_per_day(ping_period_s, sf=9):
    tsym = 2 ** sf / 125e3
    beacons = 86400 / 128
    t_beacon = airtime(sf, payload_app=17 - 13) + 0.010   # beacon frame plus 10 ms guard margin
    pings = 86400 / ping_period_s
    return beacons * (I_RX + I_MCU) * t_beacon + pings * (I_RX + I_MCU) * 8 * tsym


def main():
    rows, psi = [], []
    for sf in [7, 9, 12]:
        for t_probe in [0.010, 0.050]:
            for i_sleep in [I_SLEEP_CHIP, 10e-6, 50e-6]:
                qs = charge_per_sample(sf, t_probe)
                q_sleep = i_sleep * 86400
                q_native = NATIVE_PER_DAY * qs + q_sleep
                for name, rate, trig in [("Native 15 min", 1.0, None), ("Fixed 4 h", 0.0625, None),
                                         ("Fixed 2 h", 0.125, None), ("Event-aware, Class B 32 s", 0.106, 32),
                                         ("Event-aware, Class B 64 s", 0.106, 64), ("Event-aware, Class B 128 s", 0.106, 128)]:
                    q = rate * NATIVE_PER_DAY * qs + q_sleep + (classb_charge_per_day(trig) if trig else 0)
                    rows.append({"SF": sf, "probe_ms": t_probe * 1e3, "sleep_uA": round(i_sleep * 1e6, 2), "policy": name,
                                 "mC_per_sample": qs * 1e3, "J_per_day": q * V, "mAh_per_day": q / 3.6,
                                 "lifetime_years": CAP_AH * 3600 / q / 365.25})
                for P in [32, 64, 128]:
                    psi.append({"SF": sf, "probe_ms": t_probe * 1e3, "sleep_uA": round(i_sleep * 1e6, 2),
                                "ping_period_s": P, "classB_J_per_day": classb_charge_per_day(P) * V,
                                "native_J_per_day": q_native * V, "psi": classb_charge_per_day(P) / q_native})
    t = pd.DataFrame(rows); p = pd.DataFrame(psi)
    for df, nm in [(t, "energy_hw"), (p, "energy_psi")]:
        df.to_csv(TABLES / f"table_{nm}.csv", index=False)
        (TABLES / f"table_{nm}.md").write_text(df.to_markdown(index=False, floatfmt=".4g"))
    print("airtime SF7/9/12 (ms):", [round(airtime(s) * 1e3, 1) for s in (7, 9, 12)])
    print(t[(t.probe_ms == 10) & (t.SF == 9)].round(4).to_string(index=False))
    print(p[(p.probe_ms == 10)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

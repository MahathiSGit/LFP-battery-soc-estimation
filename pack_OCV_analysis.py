import pandas as pd
import json
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(SCRIPT_DIR, 'OX002386SA3P16S3538_cleaned.csv')
OCV_IN = os.path.join(SCRIPT_DIR, 'OCVTABLE.TXT')

df = pd.read_csv(CSV_IN)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

with open(OCV_IN) as f:
    ocv = json.load(f)

# ---------------------------------------------------------------
# Build the OCV lookup grids
# ---------------------------------------------------------------
socSteps  = ocv['socSteps']
tempSteps = [t / 10.0 for t in ocv['tempSteps']]
n_soc  = ocv['numsocSteps']
n_temp = ocv['numTempSteps']

chargeGrid    = np.zeros((n_soc, n_temp))
dischargeGrid = np.zeros((n_soc, n_temp))
for k, v in ocv.items():
    if k.startswith('ocvVal'):
        si = v['socStep'] - 1
        ti = v['tempStep'] - 1
        chargeGrid[si, ti]    = v['chargeOcv']    / 1000.0
        dischargeGrid[si, ti] = v['dischargeOcv'] / 1000.0


def bilinear(soc, temp, grid):
    soc  = min(max(soc,  socSteps[0]),  socSteps[-1])
    temp = min(max(temp, tempSteps[0]), tempSteps[-1])

    si = 0
    while si < len(socSteps) - 2 and socSteps[si + 1] < soc:
        si += 1
    s0, s1 = socSteps[si], socSteps[si + 1]
    sf = 0 if s1 == s0 else (soc - s0) / (s1 - s0)

    ti = 0
    while ti < len(tempSteps) - 2 and tempSteps[ti + 1] < temp:
        ti += 1
    t0, t1 = tempSteps[ti], tempSteps[ti + 1]
    tf = 0 if t1 == t0 else (temp - t0) / (t1 - t0)

    v00, v01 = grid[si, ti], grid[si, ti + 1]
    v10, v11 = grid[si + 1, ti], grid[si + 1, ti + 1]

    v0 = v00 * (1 - tf) + v01 * tf
    v1 = v10 * (1 - tf) + v11 * tf
    return v0 * (1 - sf) + v1 * sf


cell_cols = [f'Cell Voltage|{i}' for i in range(16)]
temp_cols = [c for c in df.columns if c.startswith('Temperatures')]

df['pack_voltage']     = df[cell_cols].sum(axis=1)
df['avg_cell_voltage'] = df[cell_cols].mean(axis=1)
df['avg_temp']         = df[temp_cols].mean(axis=1)

df['rest'] = df['Current'].abs() < 0.5
df['run']  = (df['rest'] != df['rest'].shift()).cumsum()

# ---------------------------------------------------------------
# Sign convention, verified directly against this log (not assumed):
# Current > 0 correlates with SOC increasing  -> charging
# Current < 0 correlates with SOC decreasing  -> discharging
# ---------------------------------------------------------------


def classify_by_current(rid):
    """
    Direction from the mean current of the active run immediately
    before this rest run (rid - 1). Falls back to the run immediately
    after if there's no preceding active run (rest is the very first
    run in the log). Returns None if neither side gives a usable sign.
    """
    prev = df[df['run'] == rid - 1]
    if not prev.empty:
        mean_i = prev['Current'].mean()
    else:
        nxt = df[df['run'] == rid + 1]
        if nxt.empty:
            return None
        mean_i = nxt['Current'].mean()

    if mean_i > 0:
        return 'charge'
    elif mean_i < 0:
        return 'discharge'
    return None


def classify_by_soc_trend(start_idx):
    """
    Direction from the SOC trend immediately before the rest period
    began: walks backward to the nearest row where SOC actually
    differs (SOC only updates every several rows in this log, so ties
    are skipped), then compares. Defaults to 'discharge' if no earlier
    differing SOC value exists at all (rest point at the very start of
    the log).
    """
    soc_at_rest = df.loc[start_idx, 'SOC']
    i = start_idx - 1
    while i >= 0 and df.loc[i, 'SOC'] == soc_at_rest:
        i -= 1
    if i < 0:
        return 'discharge'
    return 'charge' if soc_at_rest > df.loc[i, 'SOC'] else 'discharge'


runs = df[df['rest']].groupby('run')
records = []
for rid, g in runs:
    dur = (g['date'].max() - g['date'].min()).total_seconds() / 60
    if dur >= 10:
        last = g.iloc[-1]
        current_dir = classify_by_current(rid)
        soc_dir = classify_by_soc_trend(g.index[0])

        # Cross-check the two signals. Current sign is the more direct
        # physical measurement (SOC in this log only updates every
        # several rows, so its trend can lag); use it as primary when
        # available, and fall back to the SOC-trend classification only
        # when current gives no usable sign (e.g. no adjacent active run).
        if current_dir is not None:
            final_dir = current_dir
        else:
            final_dir = soc_dir

        agree = (current_dir == soc_dir) if current_dir is not None else None

        records.append({
            'run': rid,
            'date': last['date'],
            'rest_duration_min': round(dur, 1),
            'SOC': last['SOC'],
            'avg_temp_C': round(last['avg_temp'], 2),
            'measured_pack_V': round(last['pack_voltage'], 3),
            'measured_avg_cell_V': round(last['avg_cell_voltage'], 4),
            'direction_by_current': current_dir,
            'direction_by_soc_trend': soc_dir,
            'directions_agree': agree,
            'direction_used': final_dir,
        })

res = pd.DataFrame(records)


def estimate_ocv(row):
    if row['direction_used'] == 'charge':
        grid = chargeGrid
    elif row['direction_used'] == 'discharge':
        grid = dischargeGrid
    else:
        grid = (chargeGrid + dischargeGrid) / 2.0  # last-resort fallback only
    return round(bilinear(row['SOC'], row['avg_temp_C'], grid), 4)


res['ocv_cell_est_V'] = res.apply(estimate_ocv, axis=1)
res['ocv_pack_est_V'] = (res['ocv_cell_est_V'] * 16).round(3)
res['delta_pack_V']   = (res['measured_pack_V'] - res['ocv_pack_est_V']).round(3)
res['delta_cell_mV']  = ((res['measured_avg_cell_V'] - res['ocv_cell_est_V']) * 1000).round(1)

res = res.sort_values('date').reset_index(drop=True)
cols = ['date', 'direction_by_current', 'direction_by_soc_trend', 'directions_agree',
        'direction_used', 'SOC', 'avg_temp_C', 'rest_duration_min',
        'measured_pack_V', 'ocv_pack_est_V', 'delta_pack_V',
        'measured_avg_cell_V', 'ocv_cell_est_V', 'delta_cell_mV']
print(res[cols].to_string())

CSV_OUT = os.path.join(SCRIPT_DIR, 'pack_ocv_comparison.csv')
res[cols].to_csv(CSV_OUT, index=False)
print(f"\nSaved comparison table to {CSV_OUT}")

print()
n_disagree = (res['directions_agree'] == False).sum()
n_agree = (res['directions_agree'] == True).sum()
n_no_current_signal = res['direction_by_current'].isna().sum()
print(f"Direction agreement: {n_agree} agree, {n_disagree} disagree, "
      f"{n_no_current_signal} had no current-based signal (used SOC trend instead)")
if n_disagree > 0:
    print("\nDisagreeing rest points (worth a manual look):")
    print(res.loc[res['directions_agree'] == False, cols].to_string())

print()
print("Summary stats delta_pack_V:")
print(res['delta_pack_V'].describe())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res_sorted = res.sort_values('SOC')
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = res_sorted['direction_used'].map({'charge': '#2a78d6', 'discharge': '#c0392b'})
# mark disagreements with a black outline so they're visually distinct
edge_colors = res_sorted['directions_agree'].map({True: 'none', False: 'black', None: 'none'})
ax.scatter(res_sorted['SOC'], res_sorted['measured_pack_V'], c=colors, s=70, zorder=3,
           edgecolors=edge_colors, linewidths=1.5, label='Measured pack voltage')
ax.plot(res_sorted['SOC'], res_sorted['ocv_pack_est_V'], marker='o', linestyle='--',
        color='#eb6834', label='OCV table pack voltage (direction-aware)')
ax.set_xlabel('SOC (%) at settled rest point')
ax.set_ylabel('Pack voltage (V)')
ax.set_title('Pack-level measured voltage vs OCV table estimate\n'
              '(16S pack, current = 0, rest >= 10 min; black outline = current/SOC-trend disagreement)')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
PNG_OUT = os.path.join(SCRIPT_DIR, 'pack_ocv_comparison.png')
fig.savefig(PNG_OUT, dpi=150)
print(f"Saved plot to {PNG_OUT}")
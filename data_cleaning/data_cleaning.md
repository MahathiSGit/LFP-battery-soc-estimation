# Battery Pack Telemetry — Analysis & Cleaning

Cleans a raw BMS (battery management system) telemetry log — per-cell
voltages (16 cells), 6 temperature sensors, pack current, state of
charge, and equivalent cycle count — into a form ready for downstream
analysis (OCV comparison, SoC modeling, etc.).

## What the notebook does

1. **Loads the raw log** (`OX002386SA3P16S3538.csv`) and profiles it —
   shape, dtypes, missing values, basic stats per column.

2. **Converts units:**
   - Cell voltages: raw mV → V (÷1000)
   - Temperatures: raw 0.1 °C → °C (÷10)

3. **Parses and sorts timestamps** — `date` is stored as text in the raw
   file; this converts it to real datetimes and sorts rows
   chronologically, then checks for duplicate timestamps.

4. **Checks sampling interval** — confirms the log samples roughly every
   ~2 hours, with bursts of dense (per-second) readings during
   charge/discharge events. This is expected BMS logging behavior, not a
   data quality issue, so no action is taken here.

5. **Handles missing rows** — the first few rows in the raw file are
   missing most sensor columns (a common "cold-start" pattern: cells
   report in one at a time as the BMS polls each board). Rather than
   dropping these outright, only rows missing the *core* pack-level
   columns (`Current`, `SOC`, `Eq. Cycle`) are dropped, since those rows
   can't be used for any pack-level analysis anyway — everything else is
   kept.

6. **Visual sanity checks** — plots all 16 cell voltages and all 6
   temperature sensors over time, confirming both look smooth and
   physically plausible (no spikes, no obviously broken sensors).

7. **Reviews Current / SOC / Eq. Cycle** — confirms `Eq. Cycle` rises in
   small, mostly monotonic steps as expected for a cycle counter. Flags
   (without resolving) that `SOC` in this file ranges in the thousands
   (9000–9900), which looks more like a raw capacity/charge counter
   (e.g. mAh remaining) than a 0–100% state-of-charge percentage — this
   is noted as something to confirm against the source system's
   documentation rather than guessed at.

8. **Checks for fully duplicate rows** (excluding timestamp) — none
   found in this file.

9. **Exports the cleaned dataset** to
   `OX002386SA3P16S3538_cleaned.csv`.

## What was actually changed vs. the raw file

- Cell voltages and temperatures converted to standard units (V, °C).
- Timestamps parsed to real datetimes and sorted.
- Rows missing all core pack-level fields dropped.
- No other rows or values altered — cell-voltage spike/outlier detection
  was **intentionally left out** of this pass (see Known limitations).

## Requirements

```bash
conda env create -f environment.yml
conda activate battery-soc
```

## Running it

Open and run top to bottom in Jupyter:
```bash
jupyter notebook battery_data_analysis.ipynb
```
Requires `OX002386SA3P16S3538.csv` (the raw log) to be in the same
folder. Running it end to end produces `OX002386SA3P16S3538_cleaned.csv`
in that same folder.

## Output

- `OX002386SA3P16S3538_cleaned.csv` — the cleaned dataset, used as the
  input for the downstream OCV comparison and SoC modeling work.


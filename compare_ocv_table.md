# Pack OCV Analysis

Compares measured pack/cell voltage against an OCV lookup table, at every
settled rest point in a battery log — using the *correct* charge or
discharge OCV curve for each point instead of averaging the two.

## What it does

1. **Loads the pack log** (`OX002386SA3P16S3538_cleaned.csv`) and the OCV
   reference table (`OCVTABLE.TXT`, a JSON file despite the `.TXT`
   extension).

2. **Builds two OCV lookup grids** from the table — one for the charge
   branch, one for discharge — indexed by SoC step (0–100% in steps of
   10) and temperature step (-10, 0, 10, 25, 45 °C).

3. **Computes pack/cell voltage and average temperature** per row from
   the 16 individual cell voltage columns and the temperature sensor
   columns.

4. **Finds settled rest points** — contiguous stretches where
   `|Current| < 0.5 A` lasting at least 10 minutes. These are the only
   points where a measured voltage can be fairly compared to an
   *equilibrium* OCV table value (voltage under load includes IR drop
   and polarization that have nothing to do with OCV).

5. **Classifies each rest point as following a charge or discharge step,
   two ways, then cross-checks them:**
   - **By current sign** — looks at the mean current of the active run
     immediately before the rest period began. Positive current →
     charging, negative → discharging (verified directly against this
     log's own SoC trend, not assumed). Falls back to the run
     immediately *after* the rest if there's no preceding active run.
   - **By SoC trend** — walks backward from the start of the rest
     period to the nearest row where SoC actually differs (SoC only
     updates every several rows in this log, so ties are skipped). SoC
     rising into the rest point → charge, falling → discharge. Defaults
     to `discharge` if no earlier differing SoC value exists at all
     (rest point right at the start of the log).
   - **Cross-check:** the two are compared and logged
     (`directions_agree`). Current sign is used as the primary signal
     when available — it's a direct physical measurement, whereas SoC
     trend can lag since SoC only updates intermittently — and the
     script only falls back to the SoC-trend classification when
     current gives no usable sign at all.
   - Disagreements between the two methods are printed explicitly and
     marked with a black outline on the output plot, rather than
     silently resolved one way.

6. **Looks up the OCV estimate** for each rest point's actual SoC and
   temperature, using **bilinear interpolation** — see
   [How the interpolation works](#how-the-interpolation-works) below —
   over whichever grid (charge or discharge) matches the direction used.

7. **Compares measured vs. estimated voltage** at both the pack level
   (×16 cells) and the average single-cell level, and reports the
   difference (`delta_pack_V`, `delta_cell_mV`).

8. **Saves a CSV and a plot**:
   - `pack_ocv_comparison.csv` — one row per settled rest point, with
     both classification methods, whether they agreed, which one was
     used, and the measured/estimated voltage delta.
   - `pack_ocv_comparison.png` — measured voltage (colored by the
     direction used) against the OCV-table estimate across SoC, with
     disagreeing points outlined in black.

## How the interpolation works

`bilinear(soc, temp, grid)` estimates OCV at any (SoC, temperature) point
that falls *between* the table's fixed grid points:

1. **Clip, don't extrapolate** — SoC/temp outside the table's range are
   pulled back to the nearest table edge.
2. **Find the SoC bracket** `[s0, s1]` that contains the actual SoC, and
   the fractional position `sf` inside it (0 = at `s0`, 1 = at `s1`).
3. **Find the temperature bracket** `[t0, t1]` the same way, giving `tf`.
4. **Read the four grid corners** around that point: `v00, v01, v10,
   v11`.
5. **Interpolate along temperature first**, at each SoC edge:
   `v0 = v00*(1-tf) + v01*tf`, `v1 = v10*(1-tf) + v11*tf`.
6. **Then interpolate along SoC** between those two results:
   `result = v0*(1-sf) + v1*sf`.

Standard bilinear interpolation — interpolate one axis, then the other.

## Files needed in the same folder

- `pack_OCV_analysis.py`
- `OX002386SA3P16S3538_cleaned.csv` (your pack log)
- `OCVTABLE.TXT` (the OCV reference table)
- `environment.yml`

## Running it

```bash
conda env create -f environment.yml
conda activate battery-soc
python pack_OCV_analysis.py
```

No arguments — paths are resolved relative to the script's own location,
so it works regardless of which directory you run it from, as long as
the CSV and OCV table sit alongside the script.

## Output

Printed to console — including a direction-agreement summary and a
listing of any disagreeing rest points — and saved as:
- `pack_ocv_comparison.csv`
- `pack_ocv_comparison.png`

## Known limitations

- **Very few settled rest points.** This log only has 11 rest periods
  meeting the ≥10-minute threshold — treat the delta statistics as a
  rough sanity check, not a robust calibration.
- **The one classification disagreement found so far** is the very
  first rest point in the log: current sign says `charge`, SoC trend
  defaults to `discharge` because there's no earlier SoC value to
  compare against at the very start of the log. Current sign is treated
  as more trustworthy here since it's a real measurement rather than a
  fallback default — but it's worth a manual sanity check on that
  specific point if precision matters for your use case.
- **Table vs. pack mismatch.** If the OCV table wasn't built specifically
  for this exact cell/pack, systematic voltage offsets are expected and
  don't necessarily indicate a bug in this script — they may reflect a
  genuine table-vs-hardware mismatch worth investigating separately.
"""
Data preparation for the plateau-region GRU.

Pipeline:
  1. Load the raw pack log.
  2. Detect plateau sessions (checkpoint detector).
  3. Turn each session into a (features, target) sequence:
       features = [voltage, current, temperature, elapsed_seconds]
       target   = SoC delta relative to the session's anchor SoC
                  (anchor = SoC at the first sample of the session,
                  standing in for "the OCV-table value at plateau entry")

NOTE ON THIS DATASET: this single-cell log only contains three plateau
excursions, which is enough to validate the pipeline end-to-end but not
enough to train a GRU that will generalize. Treat this as a scaffold —
point `load_sessions` at a concatenation of many cells/logs once you have
them, and everything downstream (windowing, scaling, training) stays the
same.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


VOLTAGE_PREFIX = "Cell Voltage"
TEMP_PREFIX = "Temperatures"


@dataclass
class Session:
    cell_id: str
    features: np.ndarray   # (T, 4) -> [voltage, current, temperature, elapsed_s]
    target: np.ndarray     # (T,)   -> SoC delta from anchor
    soc_true: np.ndarray   # (T,)   -> raw SoC, kept for evaluation/plotting
    anchor_soc: float      # SoC at first sample (the "table" value at entry)


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    v_cols = [c for c in df.columns if c.startswith(VOLTAGE_PREFIX)]
    t_cols = [c for c in df.columns if c.startswith(TEMP_PREFIX)]
    df["voltage"] = df[v_cols].mean(axis=1)
    df["temperature"] = df[t_cols].mean(axis=1)
    return df


def compute_rolling_slope(df: pd.DataFrame, window: int = 5, min_dsoc: float = 0.3,
                           max_abs_current=None) -> np.ndarray:
    """
    Local dV/dSoC estimated by linear regression over a rolling window
    (more robust than a raw point-to-point diff, which is dominated by
    sample-to-sample noise on real logs).

    If `max_abs_current` is set, only windows where |current| stays below
    that value the whole way through are used -- this is what actually
    approximates dOCV/dSoC rather than a current-contaminated slope
    (terminal voltage under load includes IR drop and polarization, which
    have nothing to do with the equilibrium OCV curve). Windows that don't
    qualify are left as NaN, not silently included.
    """
    v = df["voltage"].to_numpy()
    soc = df["SOC"].to_numpy()
    cur = df["Current"].to_numpy()
    n = len(df)
    slopes = np.full(n, np.nan)
    for i in range(window, n):
        if max_abs_current is not None:
            if np.abs(cur[i - window:i + 1]).max() > max_abs_current:
                continue
        ws = soc[i - window:i + 1]
        wv = v[i - window:i + 1]
        if ws.max() - ws.min() < min_dsoc:
            continue
        A = np.vstack([ws, np.ones_like(ws)]).T
        slope, _ = np.linalg.lstsq(A, wv, rcond=None)[0]
        slopes[i] = slope
    return slopes


def flag_plateau_by_slope(df: pd.DataFrame, slope_thresh: float, window: int = 5,
                           min_dsoc: float = 0.3, max_abs_current: float = 3.0,
                           min_coverage: float = 0.5):
    """
    The real checkpoint detector: flags a sample as "plateau" when the local
    |dOCV/dSoC| falls below `slope_thresh`, estimated only from low-current
    (near-rest) windows.

    Returns None instead of a mask if coverage across the SoC range is too
    sparse to trust (< `min_coverage` fraction of 10%-wide SoC bins have a
    valid estimate) -- that's a signal to fall back to
    `flag_plateau_placeholder`, not a reason to silently return a mask
    built on almost no data.
    """
    slopes = compute_rolling_slope(df, window=window, min_dsoc=min_dsoc,
                                    max_abs_current=max_abs_current)
    valid = ~np.isnan(slopes)
    if valid.sum() > 0:
        bins_covered = pd.cut(df.loc[valid, "SOC"], bins=range(0, 101, 10)).nunique()
        coverage = bins_covered / 10
    else:
        coverage = 0.0
    if coverage < min_coverage:
        return None
    is_plateau = valid & (np.abs(slopes) < slope_thresh)
    return pd.Series(is_plateau, index=df.index)


def flag_plateau_placeholder(df: pd.DataFrame, low: float, high: float) -> pd.Series:
    """
    Fallback used ONLY when `flag_plateau_by_slope` reports insufficient
    coverage (see above) -- i.e. when there isn't enough low-current data
    across the SoC range to measure dOCV/dSoC directly, which is the
    current situation for this single-cell log.

    `low`/`high` are not guesses: pass in the knee boundaries from your
    reference OCV curve (manufacturer datasheet or literature, calibrated
    against this cell's real rest points per the earlier discussion) so the
    split is still physically grounded, just not measured from this file.
    This function exists to make that fallback explicit and swappable
    rather than a silent magic number buried in the pipeline.
    """
    return (df["SOC"] >= low) & (df["SOC"] <= high)


def extract_sessions(df: pd.DataFrame, plateau_mask: pd.Series, cell_id: str = "cell0",
                      min_len: int = 20) -> list[Session]:
    """Split contiguous plateau runs into Session objects."""
    seg_id = (plateau_mask != plateau_mask.shift()).cumsum()
    sessions = []
    for _, seg in df[plateau_mask].groupby(seg_id[plateau_mask]):
        if len(seg) < min_len:
            continue
        t0 = seg["date"].iloc[0]
        elapsed = (seg["date"] - t0).dt.total_seconds().to_numpy()
        voltage = seg["voltage"].to_numpy()
        current = seg["Current"].to_numpy()
        temperature = seg["temperature"].to_numpy()
        soc_true = seg["SOC"].to_numpy()
        anchor = soc_true[0]

        features = np.stack([voltage, current, temperature, elapsed], axis=1)
        target = soc_true - anchor
        sessions.append(Session(cell_id=cell_id, features=features, target=target,
                                 soc_true=soc_true, anchor_soc=anchor))
    return sessions


def load_sessions(csv_path: str, cell_id: str = "cell0",
                   slope_thresh: float = 0.001, slope_kwargs: dict | None = None,
                   placeholder_bounds: tuple[float, float] | None = None,
                   use_placeholder: bool = False) -> list[Session]:
    """
    Tries the real, data-driven slope=0 detector first. Falls back to the
    reference-curve placeholder bounds ONLY if `use_placeholder=True` is
    passed explicitly -- this is a deliberate, logged decision now, not a
    silent substitution. If slope detection has enough coverage to trust,
    it's used regardless of `use_placeholder`, so this never downgrades a
    cell that actually has good data.
    """
    df = load_raw(csv_path)
    mask = flag_plateau_by_slope(df, slope_thresh=slope_thresh, **(slope_kwargs or {}))
    if mask is not None:
        print(f"[{cell_id}] using slope-based checkpoint detector (data-driven).")
    elif use_placeholder:
        if placeholder_bounds is None:
            raise ValueError(f"[{cell_id}] use_placeholder=True but no placeholder_bounds given.")
        print(f"[{cell_id}] slope detection insufficient -- using placeholder bounds "
              f"{placeholder_bounds} (accepted as a deliberate, temporary stand-in "
              f"pending better low-current coverage; NOT measured from this file).")
        mask = flag_plateau_placeholder(df, *placeholder_bounds)
    else:
        raise ValueError(
            f"[{cell_id}] slope-based detection has insufficient coverage and "
            f"use_placeholder=False. Pass use_placeholder=True with placeholder_bounds "
            f"to proceed anyway."
        )
    return extract_sessions(df, mask, cell_id=cell_id)


def load_sessions_multi(csv_paths: dict[str, str], **kwargs) -> list[Session]:
    """
    Load and pool sessions from multiple cells in one call.
    csv_paths: {cell_id: path}. All other kwargs pass through to load_sessions,
    so the same placeholder/slope config applies to every cell unless you loop
    and call load_sessions individually with per-cell overrides.
    """
    all_sessions = []
    for cell_id, path in csv_paths.items():
        all_sessions.extend(load_sessions(path, cell_id=cell_id, **kwargs))
    return all_sessions


class FeatureScaler:
    """Standardizes the 4 input features using stats from a set of sessions."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sessions: list[Session]):
        all_feats = np.concatenate([s.features for s in sessions], axis=0)
        self.mean = all_feats.mean(axis=0)
        self.std = all_feats.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std
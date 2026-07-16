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
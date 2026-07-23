#!/usr/bin/env python3
"""Time-aligned BIS--heart-rate directed-dependence reanalysis.

This script preserves VitalDB timestamps, does not convert monitor-derived
heart-rate samples into beat-to-beat RR intervals, and uses a fixed
early-intraoperative epoch so that recording duration cannot mechanically
determine the exposure.

The estimand is deliberately named *linear-Gaussian directed dependence*.
For Gaussian autoregressive processes it is equivalent to transfer entropy,
but it is not interpreted as physiological or causal information flow.

Primary analysis
----------------
* Signals: BIS/BIS, Solar8000/HR, BIS/SQI.
* Epoch: first 30 min after recorded operation start.
* Alignment: 10 s timestamp bins; no compression across missing intervals.
* Exposure: row-equivalent paired-shift-corrected BIS->HR directed dependence.
* Outcome: in-hospital death or ICU stay >3 days.

Outputs
-------
* data/13_time_aligned_coupling_features.csv
* results/13_time_aligned_primary.csv
* results/13_time_aligned_summary.json
* results/13_time_aligned_validity.csv
* results/13_track_availability_validity.csv
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import time
import warnings
from pathlib import Path
from typing import Iterable

import aiohttp
import numpy as np
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
API_URL = "https://api.vitaldb.net"

TRACK_NAMES = {
    "hr": "Solar8000/HR",
    "bis": "BIS/BIS",
    "sqi": "BIS/SQI",
}

PRIMARY_DURATION_SECONDS = 30 * 60
BIN_SECONDS = 10.0
MIN_VALID_FRACTION = 0.80
MAX_CONCURRENT = 30
BATCH_SIZE = 60
N_SURROGATES = 20


def parse_track_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column VitalDB track while preserving timestamps."""
    if not text.strip():
        return np.array([], dtype=float), np.array([], dtype=float)
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    times = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame.iloc[:, -1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    times = times[valid]
    values = values[valid]
    if not len(times):
        return times, values
    order = np.argsort(times, kind="stable")
    return times[order], values[order]


def bin_track(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start: float,
    duration: float,
    bin_seconds: float,
) -> np.ndarray:
    """Median-bin a track on its true time axis without interpolating gaps."""
    n_bins = int(math.ceil(duration / bin_seconds))
    output = np.full(n_bins, np.nan, dtype=float)
    if not len(times):
        return output
    keep = (
        np.isfinite(times)
        & np.isfinite(values)
        & (times >= start)
        & (times < start + duration)
    )
    if not keep.any():
        return output
    kept_times = times[keep]
    kept_values = values[keep]
    indices = np.floor((kept_times - start) / bin_seconds).astype(int)
    for index in np.unique(indices):
        vals = kept_values[indices == index]
        output[index] = float(np.nanmedian(vals))
    return output


def extract_fixed_epoch(
    tracks: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    operation_start: float,
    duration_seconds: float = PRIMARY_DURATION_SECONDS,
    bin_seconds: float = BIN_SECONDS,
    sqi_threshold: float = 50.0,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, np.ndarray | float] | None:
    """Create a fixed operation-start epoch with simultaneous quality masks."""
    if not np.isfinite(operation_start):
        return None
    required = {"hr", "bis", "sqi"}
    if not required.issubset(tracks):
        return None

    hr = bin_track(*tracks["hr"], start=operation_start, duration=duration_seconds, bin_seconds=bin_seconds)
    bis = bin_track(*tracks["bis"], start=operation_start, duration=duration_seconds, bin_seconds=bin_seconds)
    sqi = bin_track(*tracks["sqi"], start=operation_start, duration=duration_seconds, bin_seconds=bin_seconds)

    physiologic = (
        np.isfinite(hr)
        & np.isfinite(bis)
        & np.isfinite(sqi)
        & (hr >= 30)
        & (hr <= 200)
        & (bis >= 20)
        & (bis <= 80)
        & (sqi >= sqi_threshold)
    )
    valid_fraction = float(physiologic.mean())
    if valid_fraction < min_valid_fraction:
        return None

    hr = hr.astype(float)
    bis = bis.astype(float)
    hr[~physiologic] = np.nan
    bis[~physiologic] = np.nan
    return {
        "hr": hr,
        "bis": bis,
        "sqi": sqi,
        "valid_fraction": valid_fraction,
        "n_valid_bins": int(physiologic.sum()),
    }


def _lagged_components(
    source: np.ndarray,
    target: np.ndarray,
    *,
    order: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unfiltered lagged components on one common row index."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError("source and target must have identical shapes")
    offset = order + lag - 1
    if len(source) <= offset + 10:
        return np.empty((0, order)), np.empty((0, order)), np.empty(0)

    target_future = target[offset:]
    target_past = np.column_stack(
        [target[offset - lag - j : len(target) - lag - j] for j in range(order)]
    )
    source_past = np.column_stack(
        [source[offset - lag - j : len(source) - lag - j] for j in range(order)]
    )
    return target_past, source_past, target_future


def _lagged_design(
    source: np.ndarray,
    target: np.ndarray,
    *,
    order: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_past, source_past, target_future = _lagged_components(
        source,
        target,
        order=order,
        lag=lag,
    )
    valid = (
        np.isfinite(target_future)
        & np.isfinite(target_past).all(axis=1)
        & np.isfinite(source_past).all(axis=1)
    )
    return target_past[valid], source_past[valid], target_future[valid]


def _dependence_from_design(
    target_past: np.ndarray,
    source_past: np.ndarray,
    future: np.ndarray,
    *,
    min_rows: int,
) -> float:
    """Compute conditional residual-variance dependence on fixed design rows."""
    if len(future) < min_rows:
        return float("nan")
    restricted = np.column_stack([np.ones(len(future)), target_past])
    full = np.column_stack([np.ones(len(future)), target_past, source_past])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        restricted_resid = future - restricted @ np.linalg.lstsq(
            restricted, future, rcond=None
        )[0]
        full_resid = future - full @ np.linalg.lstsq(full, future, rcond=None)[0]
    if not np.isfinite(restricted_resid).all() or not np.isfinite(full_resid).all():
        return float("nan")
    var_restricted = float(np.mean(restricted_resid**2))
    var_full = float(np.mean(full_resid**2))
    if var_restricted <= 0 or var_full <= 0:
        return float("nan")
    return float(max(0.0, 0.5 * np.log(var_restricted / var_full)))


def linear_gaussian_te(
    source: np.ndarray,
    target: np.ndarray,
    *,
    order: int = 2,
    lag: int = 1,
    min_rows: int = 80,
) -> float:
    """Gaussian conditional-variance TE without compressing missing intervals."""
    target_past, source_past, future = _lagged_design(source, target, order=order, lag=lag)
    return _dependence_from_design(
        target_past,
        source_past,
        future,
        min_rows=min_rows,
    )


def paired_shift_dependence(
    source: np.ndarray,
    target: np.ndarray,
    *,
    shift: int,
    order: int = 2,
    lag: int = 1,
    min_rows: int = 80,
) -> dict[str, float | int] | None:
    """Compare observed and shifted dependence on identical complete rows.

    Moving a source array also moves its missing-value mask.  Estimating the
    observed and shifted statistics on their separate complete-case rows can
    therefore mix finite-sample background with a change in row count.  This
    paired construction intersects both designs first and evaluates both
    statistics on the same target rows.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    observed_target_past, observed_source_past, future = _lagged_components(
        source,
        target,
        order=order,
        lag=lag,
    )
    shifted_target_past, shifted_source_past, shifted_future = _lagged_components(
        np.roll(source, int(shift)),
        target,
        order=order,
        lag=lag,
    )
    shared = (
        np.isfinite(future)
        & np.isfinite(shifted_future)
        & np.isfinite(observed_target_past).all(axis=1)
        & np.isfinite(shifted_target_past).all(axis=1)
        & np.isfinite(observed_source_past).all(axis=1)
        & np.isfinite(shifted_source_past).all(axis=1)
    )
    shared_rows = int(shared.sum())
    if shared_rows < min_rows:
        return None
    observed = _dependence_from_design(
        observed_target_past[shared],
        observed_source_past[shared],
        future[shared],
        min_rows=min_rows,
    )
    shifted = _dependence_from_design(
        shifted_target_past[shared],
        shifted_source_past[shared],
        shifted_future[shared],
        min_rows=min_rows,
    )
    if not np.isfinite(observed) or not np.isfinite(shifted):
        return None
    return {
        "observed": float(observed),
        "shifted": float(shifted),
        "difference": float(observed - shifted),
        "shared_rows": shared_rows,
        "observed_rows": shared_rows,
        "shifted_rows": shared_rows,
    }


def surrogate_correction_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    order: int = 2,
    lag: int = 1,
    n_surrogates: int = N_SURROGATES,
    seed: int = 0,
    min_rows: int = 80,
) -> dict[str, float | int]:
    """Return a row-equivalent paired circular-shift correction and audit data."""
    observed = linear_gaussian_te(
        source,
        target,
        order=order,
        lag=lag,
        min_rows=min_rows,
    )
    if not np.isfinite(observed):
        return {
            "observed": float("nan"),
            "paired_observed_median": float("nan"),
            "null_median": float("nan"),
            "corrected": float("nan"),
            "median_shared_rows": 0,
            "min_shared_rows": 0,
            "n_valid_shifts": 0,
        }
    rng = np.random.default_rng(seed)
    n = len(source)
    low = max(order + lag + 2, n // 4)
    high = max(low + 1, 3 * n // 4)
    valid = []
    attempts = 0
    max_attempts = max(100, n_surrogates * 20)
    while len(valid) < n_surrogates and attempts < max_attempts:
        shift = int(rng.integers(low, high))
        attempts += 1
        item = paired_shift_dependence(
            source,
            target,
            shift=shift,
            order=order,
            lag=lag,
            min_rows=min_rows,
        )
        if item is not None:
            valid.append(item)
    if not valid:
        return {
            "observed": float(observed),
            "paired_observed_median": float("nan"),
            "null_median": float("nan"),
            "corrected": float("nan"),
            "median_shared_rows": 0,
            "min_shared_rows": 0,
            "n_valid_shifts": 0,
        }
    observed_values = np.asarray([item["observed"] for item in valid], dtype=float)
    shifted_values = np.asarray([item["shifted"] for item in valid], dtype=float)
    differences = np.asarray([item["difference"] for item in valid], dtype=float)
    shared_rows = np.asarray([item["shared_rows"] for item in valid], dtype=float)
    return {
        "observed": float(observed),
        "paired_observed_median": float(np.median(observed_values)),
        "null_median": float(np.median(shifted_values)),
        "corrected": float(np.median(differences)),
        "median_shared_rows": float(np.median(shared_rows)),
        "min_shared_rows": int(np.min(shared_rows)),
        "n_valid_shifts": int(len(valid)),
    }


def surrogate_corrected_te(
    source: np.ndarray,
    target: np.ndarray,
    *,
    order: int = 2,
    lag: int = 1,
    n_surrogates: int = N_SURROGATES,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return observed, paired-shift null, and row-equivalent correction."""
    result = surrogate_correction_diagnostics(
        source,
        target,
        order=order,
        lag=lag,
        n_surrogates=n_surrogates,
        seed=seed,
    )
    return (
        float(result["observed"]),
        float(result["null_median"]),
        float(result["corrected"]),
    )


def low_frequency_coherence(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sampling_hz: float = 0.1,
) -> float:
    """Exploratory coherence after interpolation of at most two isolated bins."""
    frame = pd.DataFrame({"x": x, "y": y}).interpolate(limit=2, limit_area="inside")
    if frame.isna().any().any() or len(frame) < 64:
        return float("nan")
    x_values = signal.detrend(frame["x"].to_numpy(dtype=float))
    y_values = signal.detrend(frame["y"].to_numpy(dtype=float))
    frequencies, coherence = signal.coherence(
        x_values,
        y_values,
        fs=sampling_hz,
        nperseg=min(64, len(frame)),
    )
    mask = (frequencies >= 0.003) & (frequencies <= 0.04)
    return float(np.nanmean(coherence[mask])) if mask.any() else float("nan")


def compute_epoch_features(epoch: dict[str, np.ndarray | float], caseid: int) -> dict[str, float | int]:
    hr = np.asarray(epoch["hr"], dtype=float)
    bis = np.asarray(epoch["bis"], dtype=float)

    b2h = surrogate_correction_diagnostics(
        bis, hr, seed=int(caseid) * 2 + 1
    )
    h2b = surrogate_correction_diagnostics(
        hr, bis, seed=int(caseid) * 2 + 2
    )
    b2h_raw = float(b2h["observed"])
    b2h_null = float(b2h["null_median"])
    b2h_corrected = float(b2h["corrected"])
    h2b_raw = float(h2b["observed"])
    h2b_null = float(h2b["null_median"])
    h2b_corrected = float(h2b["corrected"])
    return {
        "caseid": int(caseid),
        "valid_fraction": float(epoch["valid_fraction"]),
        "n_valid_bins": int(epoch["n_valid_bins"]),
        "mean_hr": float(np.nanmean(hr)),
        "mean_bis": float(np.nanmean(bis)),
        "te_bis_to_hr_raw": b2h_raw,
        "te_bis_to_hr_null": b2h_null,
        "te_bis_to_hr_corrected": b2h_corrected,
        "te_bis_to_hr_paired_observed": float(b2h["paired_observed_median"]),
        "te_bis_to_hr_shared_rows_median": float(b2h["median_shared_rows"]),
        "te_bis_to_hr_shared_rows_min": int(b2h["min_shared_rows"]),
        "te_bis_to_hr_valid_shifts": int(b2h["n_valid_shifts"]),
        "te_hr_to_bis_raw": h2b_raw,
        "te_hr_to_bis_null": h2b_null,
        "te_hr_to_bis_corrected": h2b_corrected,
        "te_hr_to_bis_paired_observed": float(h2b["paired_observed_median"]),
        "te_hr_to_bis_shared_rows_median": float(h2b["median_shared_rows"]),
        "te_hr_to_bis_shared_rows_min": int(h2b["min_shared_rows"]),
        "te_hr_to_bis_valid_shifts": int(h2b["n_valid_shifts"]),
        "te_asymmetry_corrected": b2h_corrected - h2b_corrected,
        "te_total_corrected": b2h_corrected + h2b_corrected,
        "coherence_low": low_frequency_coherence(bis, hr),
    }


async def fetch_track(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    tid: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not tid or pd.isna(tid):
        return np.array([], dtype=float), np.array([], dtype=float)
    async with semaphore:
        try:
            async with session.get(
                f"{API_URL}/{tid}",
                timeout=aiohttp.ClientTimeout(total=90),
            ) as response:
                if response.status != 200:
                    return np.array([], dtype=float), np.array([], dtype=float)
                return parse_track_csv(await response.text())
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return np.array([], dtype=float), np.array([], dtype=float)


async def process_remote_case(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: pd.Series,
    track_ids: dict[str, str],
) -> dict[str, float | int | str]:
    tasks = {
        name: fetch_track(session, semaphore, track_ids.get(name))
        for name in TRACK_NAMES
    }
    fetched = await asyncio.gather(*tasks.values())
    tracks = dict(zip(tasks.keys(), fetched, strict=True))
    if any(len(tracks[name][0]) == 0 for name in TRACK_NAMES):
        return {"caseid": int(row.caseid), "status": "missing_track"}

    epoch = extract_fixed_epoch(tracks, operation_start=float(row.opstart))
    if epoch is None:
        return {"caseid": int(row.caseid), "status": "insufficient_fixed_epoch"}
    features = compute_epoch_features(epoch, int(row.caseid))
    features["status"] = "ok"
    return features


def build_track_lookup(track_map: pd.DataFrame) -> dict[int, dict[str, str]]:
    reverse = {track_name: short for short, track_name in TRACK_NAMES.items()}
    subset = track_map[track_map["tname"].isin(reverse)].copy()
    subset["short"] = subset["tname"].map(reverse)
    lookup: dict[int, dict[str, str]] = {}
    for row in subset.itertuples(index=False):
        lookup.setdefault(int(row.caseid), {})[row.short] = row.tid
    return lookup


def select_clinically_eligible_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Apply clinical eligibility before any waveform-availability gate."""
    cohort = cases.copy()
    cohort["anedur_min"] = (
        pd.to_numeric(cohort["aneend"], errors="coerce")
        - pd.to_numeric(cohort["anestart"], errors="coerce")
    ) / 60
    cohort["opdur_min"] = (
        pd.to_numeric(cohort["opend"], errors="coerce")
        - pd.to_numeric(cohort["opstart"], errors="coerce")
    ) / 60
    cohort["los_days"] = (
        pd.to_numeric(cohort.get("dis"), errors="coerce")
        - pd.to_numeric(cohort["opend"], errors="coerce")
    ) / 86400 if "dis" in cohort else np.nan

    cohort = cohort[
        (pd.to_numeric(cohort["age"], errors="coerce") >= 18)
        & (cohort["ane_type"] == "General")
        & (cohort["opdur_min"] >= 30)
        & cohort["asa"].notna()
    ]
    # Exclude only explicit cardiac or neurosurgical labels. Generic terms such
    # as ``tumor``, ``aneurysm`` and ``aortic`` also describe eligible
    # non-neurosurgical or vascular procedures and therefore cannot define a
    # surgical specialty on their own.
    cardiac_keywords = r"\b(?:cardiac|cabg|cpb|cardiopulmonary)\b"
    neuro_keywords = r"\b(?:neurosurgery|neurosurgical|brain|craniotomy|intracranial)\b"
    operation_text = (
        cohort["optype"].fillna("").astype(str).str.lower()
        + " "
        + cohort["opname"].fillna("").astype(str).str.lower()
    )
    cohort = cohort[
        ~operation_text.str.contains(cardiac_keywords, regex=True)
        & ~operation_text.str.contains(neuro_keywords, regex=True)
    ]
    return cohort.sort_values("caseid").reset_index(drop=True)


def select_corrected_cohort(cases: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Select clinically eligible cases with all three required signals."""
    clinical = select_clinically_eligible_cases(cases)
    required_case_sets = [
        set(tracks.loc[tracks["tname"] == track_name, "caseid"].astype(int))
        for track_name in TRACK_NAMES.values()
    ]
    signal_complete = set.intersection(*required_case_sets)
    cohort = clinical[clinical["caseid"].astype(int).isin(signal_complete)].copy()
    return cohort.sort_values("caseid").reset_index(drop=True)


def load_source_table(filename: str, endpoint: str) -> pd.DataFrame:
    """Load a repository-local VitalDB table, or retrieve it from the public API."""
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(f"{API_URL}/{endpoint}")
    table.to_csv(path, index=False)
    return table


def load_corrected_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = load_source_table("clinical_information.csv", "cases")
    all_tracks = load_source_table("track_list.csv", "trks")
    cohort = select_corrected_cohort(cases, all_tracks)
    selected_tracks = all_tracks[
        all_tracks["caseid"].astype(int).isin(set(cohort["caseid"].astype(int)))
        & all_tracks["tname"].isin(TRACK_NAMES.values())
    ].copy()
    cohort.to_csv(DATA_DIR / "13_time_aligned_cohort.csv", index=False)
    selected_tracks.to_csv(DATA_DIR / "13_time_aligned_track_list.csv", index=False)
    return cohort, selected_tracks


def _standardized_mean_difference(values: pd.Series, valid: pd.Series) -> float:
    a = pd.to_numeric(values[valid], errors="coerce").dropna()
    b = pd.to_numeric(values[~valid], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _audit_variables(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive variables shared by the two sequential selection audits."""
    frame = frame.copy()
    if "sex" in frame:
        sex = frame["sex"].astype("string")
    else:
        sex = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["sex_male"] = np.where(
        sex.isin(["M", "F"]), sex.eq("M").astype(int), np.nan
    )
    asa = _numeric_column(frame, "asa")
    frame["asa_high"] = np.where(asa.notna(), asa.ge(3).astype(int), np.nan)
    frame["emergency"] = _numeric_column(frame, "emop")
    death = _numeric_column(frame, "death_inhosp")
    icu_days = _numeric_column(frame, "icu_days")
    frame["icu_gt3"] = np.where(icu_days.notna(), icu_days.gt(3).astype(int), np.nan)
    frame["hard_outcome"] = np.where(
        death.notna() & icu_days.notna(),
        (death.eq(1) | icu_days.gt(3)).astype(int),
        np.nan,
    )
    frame["anedur_min_clean"] = _numeric_column(frame, "anedur_min").where(
        lambda value: value.between(30, 1440)
    )
    return frame


def track_availability_table(cases: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Audit selection caused by requiring all three waveform tracks."""
    frame = _audit_variables(select_clinically_eligible_cases(cases))
    required_case_sets = [
        set(tracks.loc[tracks["tname"] == track_name, "caseid"].astype(int))
        for track_name in TRACK_NAMES.values()
    ]
    signal_complete = set.intersection(*required_case_sets)
    available = frame["caseid"].astype(int).isin(signal_complete)
    rows = []
    for variable in [
        "age",
        "sex_male",
        "bmi",
        "asa",
        "asa_high",
        "emergency",
        "anedur_min_clean",
        "opdur_min",
        "death_inhosp",
        "icu_days",
        "icu_gt3",
        "hard_outcome",
    ]:
        if variable not in frame:
            continue
        values = pd.to_numeric(frame[variable], errors="coerce")
        rows.append({
            "selection_stage": "required_track_availability",
            "variable": variable,
            "available_mean": float(values[available].mean()),
            "unavailable_mean": float(values[~available].mean()),
            "standardized_mean_difference": _standardized_mean_difference(values, available),
            "available_n": int(available.sum()),
            "unavailable_n": int((~available).sum()),
            "available_nonmissing_n": int(values[available].notna().sum()),
            "unavailable_nonmissing_n": int(values[~available].notna().sum()),
        })
    return pd.DataFrame(rows)


def validity_table(cohort: pd.DataFrame, successful_caseids: set[int]) -> pd.DataFrame:
    frame = _audit_variables(cohort)
    valid = frame["caseid"].astype(int).isin(successful_caseids)
    rows = []
    for variable in [
        "age",
        "sex_male",
        "bmi",
        "asa",
        "asa_high",
        "emergency",
        "anedur_min_clean",
        "opdur_min",
        "death_inhosp",
        "icu_days",
        "icu_gt3",
        "hard_outcome",
    ]:
        if variable not in frame:
            continue
        values = pd.to_numeric(frame[variable], errors="coerce")
        rows.append({
            "selection_stage": "fixed_window_validity",
            "variable": variable,
            "valid_mean": float(values[valid].mean()),
            "invalid_mean": float(values[~valid].mean()),
            "standardized_mean_difference": _standardized_mean_difference(values, valid),
            "valid_n": int(values[valid].notna().sum()),
            "invalid_n": int(values[~valid].notna().sum()),
        })
    return pd.DataFrame(rows)


def fit_logistic_models(
    analysis: pd.DataFrame,
    exposures: list[str] | None = None,
) -> pd.DataFrame:
    """Fit case-level logistic models with patient-clustered inference.

    Baseline clinical adjustment is the primary specification.  Adding
    anesthesia duration is deliberately separated as a sensitivity model
    because duration may partly encode procedure complexity and care pathway.
    """
    analysis = analysis.copy()
    analysis["hard_composite"] = (
        (pd.to_numeric(analysis["death_inhosp"], errors="coerce") == 1)
        | (pd.to_numeric(analysis["icu_days"], errors="coerce") > 3)
    ).astype(int)
    analysis["sex_male"] = (analysis["sex"] == "M").astype(int)
    asa_numeric = pd.to_numeric(analysis["asa"], errors="coerce")
    analysis["asa_high"] = np.where(asa_numeric.notna(), (asa_numeric >= 3).astype(int), np.nan)
    analysis["emergency"] = pd.to_numeric(analysis["emop"], errors="coerce")
    analysis["anedur_clean"] = pd.to_numeric(
        analysis["anedur_min"], errors="coerce"
    ).where(lambda value: value.between(30, 1440))
    if "subjectid" in analysis:
        analysis["subject_cluster"] = analysis["subjectid"].astype("string")
        covariance_label = "cluster_subject"
    else:
        analysis["subject_cluster"] = analysis.index.astype(str)
        covariance_label = "cluster_case"

    if exposures is None:
        exposures = [
            "te_bis_to_hr_corrected",
            "te_hr_to_bis_corrected",
            "te_asymmetry_corrected",
            "te_total_corrected",
        ]
    baseline_covariates = ["age", "sex_male", "bmi", "asa_high", "emergency"]
    model_specs = [
        ("unadjusted", []),
        ("baseline_adjusted", baseline_covariates),
        ("duration_added", baseline_covariates + ["anedur_clean"]),
    ]
    rows: list[dict[str, float | int | str]] = []

    for exposure in exposures:
        for model_label, covariates in model_specs:
            columns = [exposure, "hard_composite", "subject_cluster"] + covariates
            model_data = analysis[columns].copy()
            numeric_columns = [exposure, "hard_composite"] + covariates
            model_data[numeric_columns] = model_data[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            model_data = model_data.dropna(subset=columns)
            if len(model_data) < 100 or model_data["hard_composite"].sum() < 10:
                continue
            exposure_sd = float(model_data[exposure].std(ddof=1))
            if exposure_sd <= 0:
                continue
            model_data["exposure_z"] = (
                model_data[exposure] - model_data[exposure].mean()
            ) / exposure_sd
            x_columns = ["exposure_z"] + covariates
            design_data = model_data[x_columns].copy()
            if covariates:
                for column in ["age", "bmi", "anedur_clean"]:
                    if column not in design_data:
                        continue
                    standard_deviation = float(design_data[column].std(ddof=1))
                    if standard_deviation > 0:
                        design_data[column] = (
                            design_data[column] - design_data[column].mean()
                        ) / standard_deviation
            design = sm.add_constant(design_data, has_constant="add")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fit = sm.Logit(model_data["hard_composite"], design).fit(
                    disp=False,
                    cov_type="cluster",
                    cov_kwds={"groups": model_data["subject_cluster"]},
                    maxiter=200,
                )
            coefficient = float(fit.params["exposure_z"])
            lower, upper = fit.conf_int().loc["exposure_z"].astype(float)
            rows.append({
                "outcome": "death_or_icu_gt3d",
                "exposure": exposure,
                "model": model_label,
                "n": int(len(model_data)),
                "events": int(model_data["hard_composite"].sum()),
                "odds_ratio_per_sd": float(np.exp(coefficient)),
                "ci_low": float(np.exp(lower)),
                "ci_high": float(np.exp(upper)),
                "p_value": float(fit.pvalues["exposure_z"]),
                "covariates": ",".join(covariates),
                "covariance": covariance_label,
                "n_clusters": int(model_data["subject_cluster"].nunique()),
            })
    return pd.DataFrame(rows)


def directionality_table(features: pd.DataFrame) -> pd.DataFrame:
    """Compare timestamp-aligned raw and null-corrected directionality."""
    rows: list[dict[str, float | int | str]] = []
    specifications = [
        (
            "timestamp_aligned_raw",
            "te_bis_to_hr_raw",
            "te_hr_to_bis_raw",
        ),
        (
            "timestamp_aligned_paired_shift_corrected",
            "te_bis_to_hr_corrected",
            "te_hr_to_bis_corrected",
        ),
    ]
    for label, forward, reverse in specifications:
        paired = features.dropna(subset=[forward, reverse])
        test = stats.wilcoxon(
            paired[forward], paired[reverse], zero_method="zsplit"
        )
        rows.append({
            "specification": label,
            "n": int(len(paired)),
            "median_bis_to_hr": float(paired[forward].median()),
            "median_hr_to_bis": float(paired[reverse].median()),
            "median_difference": float((paired[forward] - paired[reverse]).median()),
            "pct_bis_to_hr_greater": float((paired[forward] > paired[reverse]).mean()),
            "wilcoxon_p": float(test.pvalue),
        })
    return pd.DataFrame(rows)


def duration_bias_audit(analysis: pd.DataFrame) -> pd.DataFrame:
    """Quantify whether each coupling metric is mechanically duration-dependent."""
    analysis = analysis.copy()
    analysis["anedur_clean"] = pd.to_numeric(
        analysis["anedur_min"], errors="coerce"
    ).where(lambda value: value.between(30, 1440))
    metrics = [
        "te_bis_to_hr_raw",
        "te_hr_to_bis_raw",
        "te_bis_to_hr_corrected",
        "te_hr_to_bis_corrected",
        "te_asymmetry_corrected",
        "te_total_corrected",
    ]
    rows = []
    for metric in metrics:
        pair = analysis[[metric, "anedur_clean"]].apply(pd.to_numeric, errors="coerce").dropna()
        correlation, p_value = stats.spearmanr(pair[metric], pair["anedur_clean"])
        rows.append({
            "metric": metric,
            "n": int(len(pair)),
            "spearman_r_with_anesthesia_duration": float(correlation),
            "p_value": float(p_value),
        })

    return pd.DataFrame(rows)


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a frame to JSON-safe records without rounding small floats to zero."""
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        row: dict[str, object] = {}
        for key, value in record.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, np.generic):
                row[key] = value.item()
            else:
                row[key] = value
        rows.append(row)
    return rows


def summarize(features: pd.DataFrame, primary: pd.DataFrame, total_cases: int) -> dict[str, object]:
    ok = features[features["status"] == "ok"].copy()
    paired = ok.dropna(subset=["te_bis_to_hr_corrected", "te_hr_to_bis_corrected"])
    if len(paired):
        wilcoxon = stats.wilcoxon(
            paired["te_bis_to_hr_corrected"],
            paired["te_hr_to_bis_corrected"],
            zero_method="zsplit",
        )
        directionality = {
            "n": int(len(paired)),
            "median_bis_to_hr_corrected": float(paired["te_bis_to_hr_corrected"].median()),
            "median_hr_to_bis_corrected": float(paired["te_hr_to_bis_corrected"].median()),
            "median_asymmetry": float(paired["te_asymmetry_corrected"].median()),
            "pct_bis_to_hr_greater": float(
                (paired["te_bis_to_hr_corrected"] > paired["te_hr_to_bis_corrected"]).mean()
            ),
            "wilcoxon_p": float(wilcoxon.pvalue),
        }
    else:
        directionality = {"n": 0}

    b2h_valid = pd.to_numeric(ok["te_bis_to_hr_valid_shifts"], errors="coerce")
    h2b_valid = pd.to_numeric(ok["te_hr_to_bis_valid_shifts"], errors="coerce")
    b2h_rows = pd.to_numeric(ok["te_bis_to_hr_shared_rows_min"], errors="coerce")
    h2b_rows = pd.to_numeric(ok["te_hr_to_bis_shared_rows_min"], errors="coerce")
    paired_shift_completeness = {
        "required_valid_shifts_per_direction": int(N_SURROGATES),
        "bis_to_hr_complete_cases": int(b2h_valid.eq(N_SURROGATES).sum()),
        "hr_to_bis_complete_cases": int(h2b_valid.eq(N_SURROGATES).sum()),
        "bis_to_hr_incomplete_cases": int((~b2h_valid.eq(N_SURROGATES)).sum()),
        "hr_to_bis_incomplete_cases": int((~h2b_valid.eq(N_SURROGATES)).sum()),
        "minimum_bis_to_hr_valid_shifts": int(b2h_valid.min()),
        "minimum_hr_to_bis_valid_shifts": int(h2b_valid.min()),
        "minimum_bis_to_hr_shared_rows": int(b2h_rows.min()),
        "minimum_hr_to_bis_shared_rows": int(h2b_rows.min()),
    }

    primary_records = records_for_json(primary) if len(primary) else []
    return {
        "analysis_version": "13_time_aligned_coupling_v2",
        "estimand": "row-equivalent paired-circular-shift-corrected linear-Gaussian directed dependence",
        "signal_pair": "BIS monitor index and monitor-derived heart rate",
        "epoch": "first 30 minutes after operation start",
        "bin_seconds": BIN_SECONDS,
        "total_candidate_cases": int(total_cases),
        "valid_fixed_epoch_cases": int(len(ok)),
        "valid_fraction": float(len(ok) / total_cases) if total_cases else float("nan"),
        "directionality": directionality,
        "paired_shift_completeness": paired_shift_completeness,
        "primary_models": primary_records,
        "interpretation_boundary": (
            "Paired-shift-corrected directed statistical dependence; not beat-to-beat HRV, causal information flow, "
            "or external validation."
        ),
    }


def completed_caseids(existing: pd.DataFrame) -> set[int]:
    """Return deterministic completions while allowing transient fetch retries."""
    if existing.empty or not {"caseid", "status"}.issubset(existing):
        return set()
    completed_statuses = {"ok", "insufficient_fixed_epoch"}
    return set(
        existing.loc[
            existing["status"].isin(completed_statuses), "caseid"
        ].astype(int)
    )


async def run_download(limit: int | None = None, resume: bool = True) -> pd.DataFrame:
    cohort, track_map = load_corrected_sources()
    track_lookup = build_track_lookup(track_map)
    candidates = cohort[cohort["caseid"].astype(int).isin(track_lookup)].copy()
    candidates = candidates.dropna(subset=["opstart"]).sort_values("caseid")
    if limit is not None:
        candidates = candidates.head(limit)

    output_path = DATA_DIR / "13_time_aligned_coupling_features.csv"
    existing = pd.DataFrame()
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        done = completed_caseids(existing)
        candidates = candidates[~candidates["caseid"].astype(int).isin(done)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT,
        ssl=False,
        force_close=True,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=120)
    new_rows: list[dict[str, float | int | str]] = []
    started = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for offset in range(0, len(candidates), BATCH_SIZE):
            batch = candidates.iloc[offset : offset + BATCH_SIZE]
            tasks = [
                process_remote_case(
                    session,
                    semaphore,
                    row,
                    track_lookup.get(int(row.caseid), {}),
                )
                for row in batch.itertuples(index=False)
            ]
            # itertuples yields namedtuples, while the processor only uses attributes.
            rows = await asyncio.gather(*tasks)
            new_rows.extend(rows)
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined = combined.drop_duplicates("caseid", keep="last").sort_values("caseid")
            combined.to_csv(output_path, index=False)
            elapsed = time.time() - started
            ok_count = int((combined["status"] == "ok").sum())
            print(
                f"Processed {min(offset + BATCH_SIZE, len(candidates))}/{len(candidates)} "
                f"new cases; cumulative valid={ok_count}; elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    if output_path.exists():
        return pd.read_csv(output_path)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cases")
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing feature file")
    parser.add_argument("--analyse-only", action="store_true", help="Skip downloads and analyse saved features")
    args = parser.parse_args()

    cases = load_source_table("clinical_information.csv", "cases")
    all_tracks = load_source_table("track_list.csv", "trks")
    cohort, _ = load_corrected_sources()
    if args.analyse_only:
        features = pd.read_csv(DATA_DIR / "13_time_aligned_coupling_features.csv")
    else:
        features = asyncio.run(run_download(limit=args.limit, resume=not args.no_resume))

    features = features[
        features["caseid"].astype(int).isin(set(cohort["caseid"].astype(int)))
    ].copy()
    ok = features[features["status"] == "ok"].copy()
    analysis = cohort.merge(ok, on="caseid", how="inner")
    primary = fit_logistic_models(analysis)
    primary.to_csv(RESULTS_DIR / "13_time_aligned_primary.csv", index=False)

    analysis["te_asymmetry_raw"] = (
        analysis["te_bis_to_hr_raw"] - analysis["te_hr_to_bis_raw"]
    )
    analysis["te_total_raw"] = (
        analysis["te_bis_to_hr_raw"] + analysis["te_hr_to_bis_raw"]
    )
    sensitivity = fit_logistic_models(
        analysis,
        exposures=[
            "te_bis_to_hr_raw",
            "te_hr_to_bis_raw",
            "te_asymmetry_raw",
            "te_total_raw",
        ],
    )
    sensitivity.to_csv(RESULTS_DIR / "13_time_aligned_sensitivity.csv", index=False)

    directionality = directionality_table(ok)
    directionality.to_csv(RESULTS_DIR / "13_directionality_audit.csv", index=False)

    duration_audit = duration_bias_audit(analysis)
    duration_audit.to_csv(RESULTS_DIR / "13_duration_bias_audit.csv", index=False)

    validity = validity_table(cohort, set(ok["caseid"].astype(int)))
    validity.to_csv(RESULTS_DIR / "13_time_aligned_validity.csv", index=False)

    track_validity = track_availability_table(cases, all_tracks)
    track_validity.to_csv(
        RESULTS_DIR / "13_track_availability_validity.csv", index=False
    )

    summary = summarize(features, primary, len(cohort))
    summary["directionality_audit"] = records_for_json(directionality)
    summary["duration_bias_audit"] = records_for_json(duration_audit)
    summary["track_availability_audit"] = records_for_json(track_validity)
    summary["fixed_window_validity_audit"] = records_for_json(validity)
    with open(RESULTS_DIR / "13_time_aligned_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

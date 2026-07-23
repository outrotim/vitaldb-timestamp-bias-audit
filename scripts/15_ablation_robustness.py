#!/usr/bin/env python3
"""Same-cohort ablation and parameter robustness for Study 21.

The script addresses the main reviewable uncertainty left by the corrected
30-minute analysis: the development and corrected estimates were previously
computed in different cohorts while several processing choices changed at
once. It therefore downloads the three public VitalDB tracks used by the
analysis into a resumable local 10-second cache and evaluates four ordered
specifications in the same cases:

1. position/length aligned, variable full-operation window;
2. timestamp aligned, variable full-operation window;
3. timestamp aligned, fixed 30-minute window;
4. timestamp aligned, fixed 30-minute window, circular-shift-null corrected.

It also evaluates the declared 3 x 3 x 3 raw-estimate robustness matrix
(AR order 1/2/3, lag 10/20/30 seconds, window 20/30/60 minutes) in one shared
eligible intersection and repeats the primary corrected specification with
100 rather than 20 circular shifts. Monitor-derived heart rate remains heart
rate throughout; no beat-to-beat RR or HRV claim is made.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import aiohttp
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
CACHE = DATA / "15_signal_cache"
CACHE_MANIFEST = DATA / "15_signal_cache_manifest.csv"
BIN_SECONDS = 10.0
MIN_VALID_FRACTION = 0.80
N_SURROGATES_PRIMARY = 20
N_SURROGATES_ROBUST = 100
CORRECTION_VERSION = "paired_shared_rows_complete_v3"
BATCH_SIZE = 50
MAX_CONCURRENT = 30


def _load_time_module():
    path = PROJECT / "scripts" / "13_time_aligned_coupling.py"
    spec = importlib.util.spec_from_file_location("study21_time_aligned", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TIME = _load_time_module()


class ParameterSpec(NamedTuple):
    order: int
    lag_seconds: int
    window_minutes: int

    @property
    def label(self) -> str:
        return f"ar{self.order}_lag{self.lag_seconds}s_win{self.window_minutes}m"


def parameter_grid() -> list[ParameterSpec]:
    """Return the predeclared 27-cell robustness grid."""
    return [
        ParameterSpec(order, lag_seconds, window_minutes)
        for order in (1, 2, 3)
        for lag_seconds in (10, 20, 30)
        for window_minutes in (20, 30, 60)
    ]


def length_aligned_streams(
    hr: np.ndarray,
    bis: np.ndarray,
    sqi: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Compress signals independently, then truncate by array position.

    The returned original bin indices make the positional pairing error
    explicit: paired elements can come from different elapsed times.
    """
    hr = np.asarray(hr, dtype=float)
    bis = np.asarray(bis, dtype=float)
    sqi = np.asarray(sqi, dtype=float)
    if not (hr.shape == bis.shape == sqi.shape):
        raise ValueError("hr, bis, and sqi must have identical binned shapes")

    hr_valid = np.isfinite(hr) & (hr >= 30) & (hr <= 200)
    bis_valid = (
        np.isfinite(bis)
        & np.isfinite(sqi)
        & (bis >= 20)
        & (bis <= 80)
        & (sqi >= 50)
    )
    hr_bin = np.flatnonzero(hr_valid)
    bis_bin = np.flatnonzero(bis_valid)
    n = min(len(hr_bin), len(bis_bin))
    return {
        "hr": hr[hr_bin[:n]],
        "bis": bis[bis_bin[:n]],
        "hr_bin": hr_bin[:n],
        "bis_bin": bis_bin[:n],
        "n_valid": int(n),
    }


def timestamp_aligned_streams(
    hr: np.ndarray,
    bis: np.ndarray,
    sqi: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Apply one simultaneous physiological/SQI mask on the true time grid."""
    hr = np.asarray(hr, dtype=float)
    bis = np.asarray(bis, dtype=float)
    sqi = np.asarray(sqi, dtype=float)
    if not (hr.shape == bis.shape == sqi.shape):
        raise ValueError("hr, bis, and sqi must have identical binned shapes")
    valid = (
        np.isfinite(hr)
        & np.isfinite(bis)
        & np.isfinite(sqi)
        & (hr >= 30)
        & (hr <= 200)
        & (bis >= 20)
        & (bis <= 80)
        & (sqi >= 50)
    )
    aligned_hr = hr.copy()
    aligned_bis = bis.copy()
    aligned_hr[~valid] = np.nan
    aligned_bis[~valid] = np.nan
    return {
        "hr": aligned_hr,
        "bis": aligned_bis,
        "n_valid": int(valid.sum()),
    }


def fixed_window(
    hr: np.ndarray,
    bis: np.ndarray,
    *,
    minutes: int,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, np.ndarray | int | float] | None:
    """Take a fixed leading window while retaining all missing time bins."""
    n_bins = int(minutes * 60 / BIN_SECONDS)
    if len(hr) < n_bins or len(bis) < n_bins:
        return None
    window_hr = np.asarray(hr[:n_bins], dtype=float).copy()
    window_bis = np.asarray(bis[:n_bins], dtype=float).copy()
    valid = np.isfinite(window_hr) & np.isfinite(window_bis)
    valid_fraction = float(valid.mean())
    if valid_fraction < min_valid_fraction:
        return None
    return {
        "hr": window_hr,
        "bis": window_bis,
        "n_valid": int(valid.sum()),
        "valid_fraction": valid_fraction,
    }


def _cache_parts() -> list[Path]:
    return sorted(CACHE.glob("part_*.parquet")) if CACHE.exists() else []


def cached_caseids() -> set[int]:
    caseids: set[int] = set()
    for path in _cache_parts():
        values = pd.read_parquet(path, columns=["caseid"])["caseid"].unique()
        caseids.update(int(value) for value in values)
    return caseids


def load_cache() -> pd.DataFrame:
    parts = _cache_parts()
    if not parts:
        raise FileNotFoundError(
            "No signal cache found. Run with --download-cache before analysis."
        )
    frame = pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)
    frame = frame.drop_duplicates(["caseid", "bin_index"], keep="last")
    return frame.sort_values(["caseid", "bin_index"]).reset_index(drop=True)


def _next_part_number() -> int:
    numbers = []
    for path in _cache_parts():
        match = re.search(r"part_(\d+)\.parquet$", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _primary_caseids() -> set[int]:
    features = pd.read_csv(DATA / "13_time_aligned_coupling_features.csv")
    return set(features.loc[features["status"].eq("ok"), "caseid"].astype(int))


def anesthesia_duration_lookup() -> dict[int, float]:
    """Return cleaned anaesthesia duration for the locked eligible cohort."""
    cohort = pd.read_csv(DATA / "13_time_aligned_cohort.csv")
    duration = pd.to_numeric(cohort["anedur_min"], errors="coerce").where(
        lambda value: value.between(30, 1440)
    )
    return {
        int(caseid): float(value)
        for caseid, value in zip(cohort["caseid"], duration, strict=True)
        if np.isfinite(value)
    }


async def _fetch_binned_case(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: object,
    track_ids: dict[str, str],
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    caseid = int(getattr(row, "caseid"))
    tasks = {
        name: TIME.fetch_track(session, semaphore, track_ids.get(name))
        for name in TIME.TRACK_NAMES
    }
    fetched = await asyncio.gather(*tasks.values())
    tracks = dict(zip(tasks, fetched, strict=True))
    missing = [name for name, (times, _) in tracks.items() if len(times) == 0]
    if missing:
        return None, {
            "caseid": caseid,
            "status": "download_failed_or_empty",
            "detail": ",".join(missing),
        }

    start = float(getattr(row, "opstart"))
    end = float(getattr(row, "opend"))
    duration = end - start
    if not np.isfinite(duration) or duration < 30 * 60 or duration > 24 * 60 * 60:
        return None, {
            "caseid": caseid,
            "status": "invalid_operation_duration",
            "detail": str(duration),
        }

    binned = {
        name: TIME.bin_track(
            *tracks[name],
            start=start,
            duration=duration,
            bin_seconds=BIN_SECONDS,
        )
        for name in TIME.TRACK_NAMES
    }
    n_bins = len(binned["hr"])
    frame = pd.DataFrame(
        {
            "caseid": np.full(n_bins, caseid, dtype=np.int32),
            "bin_index": np.arange(n_bins, dtype=np.int32),
            "elapsed_seconds": np.arange(n_bins, dtype=float) * BIN_SECONDS,
            "hr": binned["hr"],
            "bis": binned["bis"],
            "sqi": binned["sqi"],
        }
    )
    return frame, {
        "caseid": caseid,
        "status": "cached",
        "detail": f"{n_bins} bins",
    }


def _write_manifest(rows: list[dict[str, object]]) -> None:
    current = pd.read_csv(CACHE_MANIFEST) if CACHE_MANIFEST.exists() else pd.DataFrame()
    updated = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    updated["timestamp_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    updated = updated.drop_duplicates("caseid", keep="last").sort_values("caseid")
    updated.to_csv(CACHE_MANIFEST, index=False)


async def download_cache() -> None:
    """Download and cache full-operation 10-second tracks, resumably."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(DATA / "13_time_aligned_cohort.csv")
    track_map = pd.read_csv(DATA / "13_time_aligned_track_list.csv")
    wanted = _primary_caseids()
    done = cached_caseids()
    candidates = cohort[
        cohort["caseid"].astype(int).isin(wanted - done)
    ].dropna(subset=["opstart", "opend"]).sort_values("caseid")
    lookup = TIME.build_track_lookup(track_map)
    if candidates.empty:
        print(f"Signal cache already covers all {len(wanted):,} primary cases.")
        return

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT,
        ssl=False,
        force_close=True,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=120)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    part_number = _next_part_number()
    started = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for offset in range(0, len(candidates), BATCH_SIZE):
            batch = candidates.iloc[offset : offset + BATCH_SIZE]
            tasks = [
                _fetch_binned_case(
                    session,
                    semaphore,
                    row,
                    lookup.get(int(row.caseid), {}),
                )
                for row in batch.itertuples(index=False)
            ]
            results = await asyncio.gather(*tasks)
            frames = [frame for frame, _ in results if frame is not None]
            statuses = [status for _, status in results]
            if frames:
                part = pd.concat(frames, ignore_index=True)
                part.to_parquet(
                    CACHE / f"part_{part_number:04d}.parquet",
                    index=False,
                    compression="zstd",
                )
                part_number += 1
            _write_manifest(statuses)
            completed = min(offset + BATCH_SIZE, len(candidates))
            failures = sum(status["status"] != "cached" for status in statuses)
            elapsed = (time.time() - started) / 60
            print(
                f"Cached {completed}/{len(candidates)} pending cases; "
                f"batch failures={failures}; elapsed={elapsed:.1f} min",
                flush=True,
            )


def _pair_metrics(
    bis: np.ndarray,
    hr: np.ndarray,
    *,
    order: int = 2,
    lag: int = 1,
    n_surrogates: int = 0,
    caseid: int,
) -> dict[str, float]:
    if n_surrogates:
        forward_diagnostics = TIME.surrogate_correction_diagnostics(
            bis,
            hr,
            order=order,
            lag=lag,
            n_surrogates=n_surrogates,
            seed=caseid * 2 + 1,
        )
        reverse_diagnostics = TIME.surrogate_correction_diagnostics(
            hr,
            bis,
            order=order,
            lag=lag,
            n_surrogates=n_surrogates,
            seed=caseid * 2 + 2,
        )
        forward_raw = float(forward_diagnostics["observed"])
        forward_null = float(forward_diagnostics["null_median"])
        forward = float(forward_diagnostics["corrected"])
        reverse_raw = float(reverse_diagnostics["observed"])
        reverse_null = float(reverse_diagnostics["null_median"])
        reverse = float(reverse_diagnostics["corrected"])
    else:
        forward_raw = TIME.linear_gaussian_te(bis, hr, order=order, lag=lag)
        reverse_raw = TIME.linear_gaussian_te(hr, bis, order=order, lag=lag)
        forward_null = reverse_null = float("nan")
        forward, reverse = forward_raw, reverse_raw
        forward_diagnostics = reverse_diagnostics = {
            "paired_observed_median": float("nan"),
            "median_shared_rows": float("nan"),
            "min_shared_rows": float("nan"),
            "n_valid_shifts": 0,
        }
    return {
        "bis_to_hr": float(forward),
        "hr_to_bis": float(reverse),
        "asymmetry": float(forward - reverse),
        "total": float(forward + reverse),
        "bis_to_hr_raw": float(forward_raw),
        "hr_to_bis_raw": float(reverse_raw),
        "bis_to_hr_null": float(forward_null),
        "hr_to_bis_null": float(reverse_null),
        "bis_to_hr_paired_observed": float(
            forward_diagnostics["paired_observed_median"]
        ),
        "hr_to_bis_paired_observed": float(
            reverse_diagnostics["paired_observed_median"]
        ),
        "bis_to_hr_shared_rows_median": float(
            forward_diagnostics["median_shared_rows"]
        ),
        "hr_to_bis_shared_rows_median": float(
            reverse_diagnostics["median_shared_rows"]
        ),
        "bis_to_hr_shared_rows_min": float(
            forward_diagnostics["min_shared_rows"]
        ),
        "hr_to_bis_shared_rows_min": float(
            reverse_diagnostics["min_shared_rows"]
        ),
        "bis_to_hr_valid_shifts": int(forward_diagnostics["n_valid_shifts"]),
        "hr_to_bis_valid_shifts": int(reverse_diagnostics["n_valid_shifts"]),
    }


def _case_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = group.sort_values("bin_index")
    return (
        ordered["hr"].to_numpy(dtype=float),
        ordered["bis"].to_numpy(dtype=float),
        ordered["sqi"].to_numpy(dtype=float),
    )


def compute_ablation(cache: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = cache["caseid"].nunique()
    durations = anesthesia_duration_lookup()
    for number, (caseid_value, group) in enumerate(cache.groupby("caseid", sort=True), 1):
        caseid = int(caseid_value)
        hr, bis, sqi = _case_arrays(group)
        positional = length_aligned_streams(hr, bis, sqi)
        aligned = timestamp_aligned_streams(hr, bis, sqi)
        window30 = fixed_window(
            np.asarray(aligned["hr"]),
            np.asarray(aligned["bis"]),
            minutes=30,
        )
        specifications: list[tuple[str, np.ndarray, np.ndarray, int]] = [
            (
                "length_aligned_full_raw",
                np.asarray(positional["bis"]),
                np.asarray(positional["hr"]),
                int(positional["n_valid"]),
            ),
            (
                "timestamp_aligned_full_raw",
                np.asarray(aligned["bis"]),
                np.asarray(aligned["hr"]),
                int(aligned["n_valid"]),
            ),
        ]
        if window30 is not None:
            specifications.extend(
                [
                    (
                        "timestamp_aligned_30m_raw",
                        np.asarray(window30["bis"]),
                        np.asarray(window30["hr"]),
                        int(window30["n_valid"]),
                    ),
                    (
                        "timestamp_aligned_30m_shift20",
                        np.asarray(window30["bis"]),
                        np.asarray(window30["hr"]),
                        int(window30["n_valid"]),
                    ),
                ]
            )
        for label, source, target, n_valid in specifications:
            metrics = _pair_metrics(
                source,
                target,
                n_surrogates=N_SURROGATES_PRIMARY if label.endswith("shift20") else 0,
                caseid=caseid,
            )
            rows.append(
                {
                    "caseid": caseid,
                    "specification": label,
                    "anesthesia_duration_minutes": durations.get(caseid, float("nan")),
                    "n_valid_bins": n_valid,
                    **metrics,
                }
            )
        if number % 200 == 0 or number == total:
            print(f"Ablation: {number}/{total} cases", flush=True)
    return pd.DataFrame(rows)


def _complete_intersection(frame: pd.DataFrame, expected: int) -> set[int]:
    finite = frame[np.isfinite(frame["bis_to_hr"]) & np.isfinite(frame["hr_to_bis"])]
    counts = finite.groupby("caseid")["specification"].nunique()
    return set(counts[counts.eq(expected)].index.astype(int))


def _pairwise_spearman(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Spearman correlation after pairwise finite-value filtering."""
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(pd.Series(x), errors="coerce"),
            "y": pd.to_numeric(pd.Series(y), errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan"), float("nan")
    result = stats.spearmanr(pair["x"], pair["y"])
    return float(result.statistic), float(result.pvalue)


def _direction_summary(frame: pd.DataFrame, intersection: set[int]) -> pd.DataFrame:
    rows = []
    scoped = frame[frame["caseid"].astype(int).isin(intersection)]
    for specification, group in scoped.groupby("specification", sort=False):
        test = stats.wilcoxon(
            group["bis_to_hr"], group["hr_to_bis"], zero_method="zsplit"
        )
        duration_r, duration_p = _pairwise_spearman(
            group["asymmetry"], group["anesthesia_duration_minutes"]
        )
        rows.append(
            {
                "specification": specification,
                "n_same_cohort": int(len(group)),
                "median_bis_to_hr": float(group["bis_to_hr"].median()),
                "median_hr_to_bis": float(group["hr_to_bis"].median()),
                "median_difference": float(group["asymmetry"].median()),
                "pct_bis_to_hr_greater": float(
                    (group["bis_to_hr"] > group["hr_to_bis"]).mean()
                ),
                "wilcoxon_p": float(test.pvalue),
                "spearman_asymmetry_with_duration": float(duration_r),
                "duration_p": float(duration_p),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["wilcoxon_p_holm"] = multipletests(
            result["wilcoxon_p"], method="holm"
        )[1]
        result["duration_p_holm"] = multipletests(
            result["duration_p"], method="holm"
        )[1]
    return result


def adjacent_workflow_contrasts(
    ablation: pd.DataFrame,
    intersection: set[int],
    cohort: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260722,
) -> pd.DataFrame:
    """Quantify each adjacent workflow change with patient-cluster bootstrap CIs."""
    ordered = [
        "length_aligned_full_raw",
        "timestamp_aligned_full_raw",
        "timestamp_aligned_30m_raw",
        "timestamp_aligned_30m_shift20",
    ]
    scoped = ablation[ablation["caseid"].astype(int).isin(intersection)]
    wide = scoped.pivot(index="caseid", columns="specification", values="asymmetry")
    wide = wide.dropna(subset=ordered).reset_index()
    cluster_map = cohort[["caseid"]].copy()
    if "subjectid" in cohort:
        cluster_map["subject_cluster"] = cohort["subjectid"].astype("string")
        missing = cluster_map["subject_cluster"].isna()
        cluster_map.loc[missing, "subject_cluster"] = (
            "case_" + cluster_map.loc[missing, "caseid"].astype(str)
        )
    else:
        cluster_map["subject_cluster"] = "case_" + cluster_map["caseid"].astype(str)
    wide = wide.merge(cluster_map, on="caseid", how="left")
    wide["subject_cluster"] = wide["subject_cluster"].fillna(
        "case_" + wide["caseid"].astype(str)
    )

    rng = np.random.default_rng(seed)
    clusters = wide["subject_cluster"].drop_duplicates().to_numpy()
    cluster_rows = {
        cluster: np.flatnonzero(wide["subject_cluster"].to_numpy() == cluster)
        for cluster in clusters
    }
    contrasts = [
        ("alignment", ordered[0], ordered[1]),
        ("window_control", ordered[1], ordered[2]),
        ("paired_shift_correction", ordered[2], ordered[3]),
    ]
    rows = []
    for label, before_name, after_name in contrasts:
        before = wide[before_name].to_numpy(dtype=float)
        after = wide[after_name].to_numpy(dtype=float)
        delta = after - before
        absolute_delta = np.abs(delta)
        flips = (np.sign(before) * np.sign(after) < 0).astype(float)
        bootstrap = np.empty((n_bootstrap, 3), dtype=float)
        for iteration in range(n_bootstrap):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            indices = np.concatenate([cluster_rows[cluster] for cluster in sampled])
            bootstrap[iteration] = (
                np.median(delta[indices]),
                np.median(absolute_delta[indices]),
                np.mean(flips[indices]),
            )
        low = np.percentile(bootstrap, 2.5, axis=0)
        high = np.percentile(bootstrap, 97.5, axis=0)
        rows.append({
            "contrast": label,
            "from_specification": before_name,
            "to_specification": after_name,
            "n_cases": int(len(wide)),
            "n_subjects": int(len(clusters)),
            "median_signed_delta": float(np.median(delta)),
            "median_signed_delta_ci_low": float(low[0]),
            "median_signed_delta_ci_high": float(high[0]),
            "median_absolute_delta": float(np.median(absolute_delta)),
            "median_absolute_delta_ci_low": float(max(0.0, low[1])),
            "median_absolute_delta_ci_high": float(max(0.0, high[1])),
            "direction_flip_fraction": float(np.mean(flips)),
            "direction_flip_ci_low": float(max(0.0, low[2])),
            "direction_flip_ci_high": float(min(1.0, high[2])),
            "bootstrap_clusters": "subjectid",
            "bootstrap_replicates": int(n_bootstrap),
        })
    return pd.DataFrame(rows)


def _reproduction_check(ablation: pd.DataFrame) -> pd.DataFrame:
    locked = pd.read_csv(DATA / "13_time_aligned_coupling_features.csv")
    locked = locked[locked["status"].eq("ok")]
    mappings = [
        (
            "timestamp_aligned_30m_raw",
            "bis_to_hr",
            "te_bis_to_hr_raw",
        ),
        (
            "timestamp_aligned_30m_raw",
            "hr_to_bis",
            "te_hr_to_bis_raw",
        ),
        (
            "timestamp_aligned_30m_shift20",
            "bis_to_hr",
            "te_bis_to_hr_corrected",
        ),
        (
            "timestamp_aligned_30m_shift20",
            "hr_to_bis",
            "te_hr_to_bis_corrected",
        ),
    ]
    rows = []
    for specification, current_column, locked_column in mappings:
        current = ablation[ablation["specification"].eq(specification)][
            ["caseid", current_column]
        ]
        pair = current.merge(locked[["caseid", locked_column]], on="caseid")
        difference = pair[current_column] - pair[locked_column]
        rows.append(
            {
                "specification": specification,
                "current_metric": current_column,
                "locked_metric": locked_column,
                "n_compared": int(len(pair)),
                "max_absolute_difference": float(difference.abs().max()),
                "median_absolute_difference": float(difference.abs().median()),
            }
        )
    return pd.DataFrame(rows)


def sync_locked_feature_file(ablation: pd.DataFrame) -> None:
    """Replace the locked 30-minute correction with the audited paired version."""
    output = DATA / "13_time_aligned_coupling_features.csv"
    locked = pd.read_csv(output)
    raw = ablation[ablation["specification"].eq("timestamp_aligned_30m_raw")]
    corrected = ablation[
        ablation["specification"].eq("timestamp_aligned_30m_shift20")
    ]
    raw = raw.set_index("caseid")
    corrected = corrected.set_index("caseid")
    mappings = {
        "te_bis_to_hr_raw": (raw, "bis_to_hr"),
        "te_hr_to_bis_raw": (raw, "hr_to_bis"),
        "te_bis_to_hr_null": (corrected, "bis_to_hr_null"),
        "te_hr_to_bis_null": (corrected, "hr_to_bis_null"),
        "te_bis_to_hr_corrected": (corrected, "bis_to_hr"),
        "te_hr_to_bis_corrected": (corrected, "hr_to_bis"),
        "te_bis_to_hr_paired_observed": (corrected, "bis_to_hr_paired_observed"),
        "te_hr_to_bis_paired_observed": (corrected, "hr_to_bis_paired_observed"),
        "te_bis_to_hr_shared_rows_median": (corrected, "bis_to_hr_shared_rows_median"),
        "te_hr_to_bis_shared_rows_median": (corrected, "hr_to_bis_shared_rows_median"),
        "te_bis_to_hr_shared_rows_min": (corrected, "bis_to_hr_shared_rows_min"),
        "te_hr_to_bis_shared_rows_min": (corrected, "hr_to_bis_shared_rows_min"),
        "te_bis_to_hr_valid_shifts": (corrected, "bis_to_hr_valid_shifts"),
        "te_hr_to_bis_valid_shifts": (corrected, "hr_to_bis_valid_shifts"),
    }
    ok = locked["status"].eq("ok")
    for target, (source_frame, source_column) in mappings.items():
        locked.loc[ok, target] = locked.loc[ok, "caseid"].map(
            source_frame[source_column]
        )
    locked.loc[ok, "te_asymmetry_corrected"] = (
        locked.loc[ok, "te_bis_to_hr_corrected"]
        - locked.loc[ok, "te_hr_to_bis_corrected"]
    )
    locked.loc[ok, "te_total_corrected"] = (
        locked.loc[ok, "te_bis_to_hr_corrected"]
        + locked.loc[ok, "te_hr_to_bis_corrected"]
    )
    locked.loc[ok, "correction_version"] = CORRECTION_VERSION
    locked.to_csv(output, index=False)


def compute_parameter_matrix(cache: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    grid = parameter_grid()
    eligible: dict[int, dict[int, dict[str, np.ndarray | int | float]]] = {}
    total = cache["caseid"].nunique()
    for number, (caseid_value, group) in enumerate(cache.groupby("caseid", sort=True), 1):
        caseid = int(caseid_value)
        hr, bis, sqi = _case_arrays(group)
        aligned = timestamp_aligned_streams(hr, bis, sqi)
        windows = {
            minutes: fixed_window(
                np.asarray(aligned["hr"]),
                np.asarray(aligned["bis"]),
                minutes=minutes,
            )
            for minutes in (20, 30, 60)
        }
        if all(window is not None for window in windows.values()):
            eligible[caseid] = windows  # type: ignore[assignment]
        if number % 500 == 0 or number == total:
            print(f"Matrix eligibility: {number}/{total} cases", flush=True)

    durations = anesthesia_duration_lookup()
    rows: list[dict[str, object]] = []
    for number, (caseid, windows) in enumerate(eligible.items(), 1):
        duration = durations.get(caseid, float("nan"))
        for item in grid:
            window = windows[item.window_minutes]
            metrics = _pair_metrics(
                np.asarray(window["bis"]),
                np.asarray(window["hr"]),
                order=item.order,
                lag=item.lag_seconds // int(BIN_SECONDS),
                caseid=caseid,
            )
            rows.append(
                {
                    "caseid": caseid,
                    "specification": item.label,
                    "ar_order": item.order,
                    "lag_seconds": item.lag_seconds,
                    "window_minutes": item.window_minutes,
                    "anesthesia_duration_minutes": duration,
                    "n_valid_bins": int(window["n_valid"]),
                    **metrics,
                }
            )
        if number % 100 == 0 or number == len(eligible):
            print(f"Parameter matrix: {number}/{len(eligible)} cases", flush=True)
    frame = pd.DataFrame(rows)
    intersection = _complete_intersection(frame, len(grid))
    return frame[frame["caseid"].isin(intersection)].copy(), intersection


def _parameter_summary(frame: pd.DataFrame, intersection: set[int]) -> pd.DataFrame:
    cohort = pd.read_csv(DATA / "13_time_aligned_cohort.csv")
    rows = []
    for specification, group in frame.groupby("specification", sort=False):
        group = group[group["caseid"].isin(intersection)]
        test = stats.wilcoxon(
            group["bis_to_hr"], group["hr_to_bis"], zero_method="zsplit"
        )
        duration_r, duration_p = _pairwise_spearman(
            group["asymmetry"], group["anesthesia_duration_minutes"]
        )
        analysis = cohort.merge(
            group[["caseid", "bis_to_hr"]].rename(columns={"bis_to_hr": "dependence"}),
            on="caseid",
            how="inner",
        )
        model = TIME.fit_logistic_models(analysis, exposures=["dependence"])
        adjusted = model[model["model"].eq("baseline_adjusted")].iloc[0]
        first = group.iloc[0]
        rows.append(
            {
                "specification": specification,
                "ar_order": int(first["ar_order"]),
                "lag_seconds": int(first["lag_seconds"]),
                "window_minutes": int(first["window_minutes"]),
                "n_same_cohort": int(len(group)),
                "median_difference": float(group["asymmetry"].median()),
                "pct_bis_to_hr_greater": float(
                    (group["bis_to_hr"] > group["hr_to_bis"]).mean()
                ),
                "wilcoxon_p": float(test.pvalue),
                "spearman_asymmetry_with_duration": float(duration_r),
                "duration_p": float(duration_p),
                "hard_outcome_events": int(adjusted["events"]),
                "adjusted_or_per_sd": float(adjusted["odds_ratio_per_sd"]),
                "ci_low": float(adjusted["ci_low"]),
                "ci_high": float(adjusted["ci_high"]),
                "outcome_p": float(adjusted["p_value"]),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        for column in ["wilcoxon_p", "duration_p", "outcome_p"]:
            result[f"{column}_holm"] = multipletests(
                result[column], method="holm"
            )[1]
    return result


def compute_surrogate100(cache: pd.DataFrame) -> pd.DataFrame:
    output = RESULTS / "15_surrogate100_casewise.csv"
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame()
    if len(existing) and "correction_version" in existing:
        current = existing["correction_version"].eq(CORRECTION_VERSION)
        done = set(existing.loc[current, "caseid"].astype(int))
    else:
        done = set()
    rows: list[dict[str, object]] = []
    groups = [(int(caseid), group) for caseid, group in cache.groupby("caseid", sort=True)]
    pending = [(caseid, group) for caseid, group in groups if caseid not in done]
    for number, (caseid, group) in enumerate(pending, 1):
        hr, bis, sqi = _case_arrays(group)
        aligned = timestamp_aligned_streams(hr, bis, sqi)
        window = fixed_window(
            np.asarray(aligned["hr"]),
            np.asarray(aligned["bis"]),
            minutes=30,
        )
        if window is not None:
            metrics = _pair_metrics(
                np.asarray(window["bis"]),
                np.asarray(window["hr"]),
                order=2,
                lag=1,
                n_surrogates=N_SURROGATES_ROBUST,
                caseid=caseid,
            )
            rows.append({
                "caseid": caseid,
                "correction_version": CORRECTION_VERSION,
                **metrics,
            })
        if number % 100 == 0 or number == len(pending):
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            combined = combined.drop_duplicates("caseid", keep="last").sort_values("caseid")
            combined.to_csv(output, index=False)
            print(f"100-shift check: {number}/{len(pending)} pending cases", flush=True)
    return pd.read_csv(output) if output.exists() else existing


def _surrogate100_summary(frame: pd.DataFrame) -> dict[str, object]:
    locked = pd.read_csv(DATA / "13_time_aligned_coupling_features.csv")
    locked = locked[locked["status"].eq("ok")]
    pair = frame.merge(
        locked[
            [
                "caseid",
                "te_bis_to_hr_corrected",
                "te_hr_to_bis_corrected",
            ]
        ],
        on="caseid",
    )
    cohort = pd.read_csv(DATA / "13_time_aligned_cohort.csv")
    analysis = cohort.merge(
        frame[["caseid", "bis_to_hr"]].rename(columns={"bis_to_hr": "dependence"}),
        on="caseid",
    )
    model = TIME.fit_logistic_models(analysis, exposures=["dependence"])
    adjusted = model[model["model"].eq("baseline_adjusted")].iloc[0]
    correlation, correlation_p = stats.spearmanr(
        pair["bis_to_hr"], pair["te_bis_to_hr_corrected"]
    )
    return {
        "n": int(len(pair)),
        "median_bis_to_hr_100_shift": float(frame["bis_to_hr"].median()),
        "median_hr_to_bis_100_shift": float(frame["hr_to_bis"].median()),
        "median_asymmetry_100_shift": float(frame["asymmetry"].median()),
        "spearman_100_vs_20_bis_to_hr": float(correlation),
        "spearman_p": float(correlation_p),
        "adjusted_hard_outcome_or_per_sd": float(adjusted["odds_ratio_per_sd"]),
        "ci_low": float(adjusted["ci_low"]),
        "ci_high": float(adjusted["ci_high"]),
        "p_value": float(adjusted["p_value"]),
    }


def save_misalignment_example(cache: pd.DataFrame) -> None:
    """Save one deterministic case with the largest median positional offset."""
    candidates = []
    for caseid_value, group in cache.groupby("caseid", sort=True):
        hr, bis, sqi = _case_arrays(group)
        positional = length_aligned_streams(hr, bis, sqi)
        if int(positional["n_valid"]) < 100:
            continue
        offsets = np.abs(
            np.asarray(positional["hr_bin"]) - np.asarray(positional["bis_bin"])
        )
        candidates.append((float(np.median(offsets)), int(caseid_value), group, positional))
    if not candidates:
        return
    _, caseid, group, positional = max(candidates, key=lambda item: (item[0], -item[1]))
    n = min(300, int(positional["n_valid"]))
    example = pd.DataFrame(
        {
            "caseid": caseid,
            "pair_position": np.arange(n),
            "hr_bin": np.asarray(positional["hr_bin"])[:n],
            "bis_bin": np.asarray(positional["bis_bin"])[:n],
            "hr_elapsed_seconds": np.asarray(positional["hr_bin"])[:n] * BIN_SECONDS,
            "bis_elapsed_seconds": np.asarray(positional["bis_bin"])[:n] * BIN_SECONDS,
            "hr": np.asarray(positional["hr"])[:n],
            "bis": np.asarray(positional["bis"])[:n],
        }
    )
    example.to_csv(DATA / "15_example_misalignment.csv", index=False)


def analyse_cache(run_surrogate100: bool) -> None:
    cache = load_cache()
    expected = _primary_caseids()
    available = set(cache["caseid"].astype(int).unique())
    missing = expected - available
    if missing:
        raise RuntimeError(
            f"Cache is incomplete: {len(missing)} of {len(expected)} primary cases missing. "
            "Re-run --download-cache before analysis."
        )

    ablation = compute_ablation(cache)
    ablation.to_csv(RESULTS / "15_same_cohort_ablation_casewise.csv", index=False)
    intersection = _complete_intersection(ablation, expected=4)
    ablation_summary = _direction_summary(ablation, intersection)
    ablation_summary.to_csv(RESULTS / "15_same_cohort_ablation_summary.csv", index=False)
    cohort = pd.read_csv(DATA / "13_time_aligned_cohort.csv")
    adjacent = adjacent_workflow_contrasts(ablation, intersection, cohort)
    adjacent.to_csv(RESULTS / "15_adjacent_workflow_contrasts.csv", index=False)
    sync_locked_feature_file(ablation)
    reproduction = _reproduction_check(ablation)
    reproduction.to_csv(RESULTS / "15_locked_result_reproduction.csv", index=False)

    matrix, matrix_intersection = compute_parameter_matrix(cache)
    matrix.to_csv(RESULTS / "15_parameter_robustness_casewise.csv", index=False)
    matrix_summary = _parameter_summary(matrix, matrix_intersection)
    matrix_summary.to_csv(RESULTS / "15_parameter_robustness_summary.csv", index=False)
    save_misalignment_example(cache)

    surrogate_summary: dict[str, object] | None = None
    if run_surrogate100:
        surrogate = compute_surrogate100(cache)
        surrogate_summary = _surrogate100_summary(surrogate)

    payload = {
        "analysis_version": "15_same_cohort_ablation_v2",
        "cache_case_count": int(len(available)),
        "four_step_intersection_n": int(len(intersection)),
        "parameter_matrix_intersection_n": int(len(matrix_intersection)),
        "ablation": TIME.records_for_json(ablation_summary),
        "adjacent_workflow_contrasts": TIME.records_for_json(adjacent),
        "reproduction_check": TIME.records_for_json(reproduction),
        "parameter_matrix": TIME.records_for_json(matrix_summary),
        "surrogate100": surrogate_summary,
        "interpretation_boundary": (
            "A processing-bias audit of monitor BIS and monitor HR; estimates are "
            "linear-Gaussian directed dependence, not HRV or physiological causation."
        ),
    }
    with open(RESULTS / "15_ablation_robustness_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-cache",
        action="store_true",
        help="Download/resume full-operation VitalDB tracks into local Parquet parts",
    )
    parser.add_argument(
        "--analyse",
        action="store_true",
        help="Run same-cohort ablation and the 27-cell robustness matrix",
    )
    parser.add_argument(
        "--surrogate100",
        action="store_true",
        help="With --analyse, repeat the primary correction using 100 shifts",
    )
    args = parser.parse_args()
    if not args.download_cache and not args.analyse:
        parser.error("select --download-cache and/or --analyse")
    if args.download_cache:
        asyncio.run(download_cache())
    if args.analyse:
        analyse_cache(run_surrogate100=args.surrogate100)


if __name__ == "__main__":
    main()

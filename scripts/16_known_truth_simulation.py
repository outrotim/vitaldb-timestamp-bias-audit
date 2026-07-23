#!/usr/bin/env python3
"""Known-truth simulation for the Study 21 processing-bias audit.

The data-generating process contains one declared direction: a BIS-like AR(1)
process contributes to the next HR-like value, while HR does not contribute to
future BIS. Four observation scenarios then add unequal missingness, delayed
monitor start, and variable record length. The same four processing workflows
used in the empirical ablation are applied without tuning to significance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
SCENARIOS = (
    "clean_equal",
    "unequal_missingness",
    "delayed_start",
    "combined_variable",
)


def _load_ablation_module():
    path = PROJECT / "scripts" / "15_ablation_robustness.py"
    spec = importlib.util.spec_from_file_location("study21_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ABLATION = _load_ablation_module()
TIME = ABLATION.TIME


def _missing_block(values: np.ndarray, start: int, width: int) -> None:
    stop = min(len(values), start + width)
    if start < stop:
        values[start:stop] = np.nan


def simulate_record(
    *,
    seed: int,
    n_bins: int,
    scenario: str,
) -> dict[str, np.ndarray]:
    """Simulate one known BIS->HR record and its observed monitor streams."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if n_bins < 180:
        raise ValueError("n_bins must cover at least 30 minutes")
    rng = np.random.default_rng(seed)
    latent_bis = np.zeros(n_bins, dtype=float)
    latent_hr = np.zeros(n_bins, dtype=float)
    for index in range(2, n_bins):
        latent_bis[index] = 0.72 * latent_bis[index - 1] + rng.normal(scale=0.65)
        latent_hr[index] = (
            0.52 * latent_hr[index - 1]
            + 0.62 * latent_bis[index - 1]
            + rng.normal(scale=0.70)
        )

    truth_bis = np.clip(48 + 6 * latent_bis, 21, 79)
    truth_hr = np.clip(72 + 7 * latent_hr, 31, 199)
    observed_bis = truth_bis.copy()
    observed_hr = truth_hr.copy()
    sqi = np.full(n_bins, 92.0)

    if scenario in {"unequal_missingness", "combined_variable"}:
        observed_hr[rng.random(n_bins) < 0.035] = np.nan
        observed_bis[rng.random(n_bins) < 0.015] = np.nan
        sqi[rng.random(n_bins) < 0.015] = 35.0
        first = int(rng.integers(45, max(46, n_bins // 3)))
        second = int(rng.integers(max(first + 50, n_bins // 2), max(first + 51, n_bins - 30)))
        _missing_block(observed_hr, first, max(4, n_bins // 60))
        _missing_block(observed_bis, second, max(3, n_bins // 90))

    if scenario in {"delayed_start", "combined_variable"}:
        delay = 6 if scenario == "delayed_start" else int(rng.integers(3, 13))
        observed_hr[:delay] = np.nan

    if scenario == "combined_variable":
        hr_low = min(n_bins, max(180, int(n_bins * 0.75)))
        bis_low = min(n_bins, max(180, int(n_bins * 0.70)))
        hr_end = int(rng.integers(hr_low, n_bins + 1))
        bis_end = int(rng.integers(bis_low, n_bins + 1))
        observed_hr[hr_end:] = np.nan
        observed_bis[bis_end:] = np.nan
        sqi[bis_end:] = np.nan

    return {
        "bis": observed_bis,
        "hr": observed_hr,
        "sqi": sqi,
        "truth_bis": truth_bis,
        "truth_hr": truth_hr,
    }


def _workflow_rows(
    record: dict[str, np.ndarray],
    *,
    scenario: str,
    case_number: int,
) -> list[dict[str, object]]:
    positional = ABLATION.length_aligned_streams(
        record["hr"], record["bis"], record["sqi"]
    )
    aligned = ABLATION.timestamp_aligned_streams(
        record["hr"], record["bis"], record["sqi"]
    )
    window = ABLATION.fixed_window(
        np.asarray(aligned["hr"]),
        np.asarray(aligned["bis"]),
        minutes=30,
    )
    truth = ABLATION._pair_metrics(
        record["truth_bis"], record["truth_hr"], caseid=case_number
    )
    specifications: list[tuple[str, np.ndarray, np.ndarray, int]] = [
        (
            "length_aligned_full_raw",
            np.asarray(positional["bis"]),
            np.asarray(positional["hr"]),
            0,
        ),
        (
            "timestamp_aligned_full_raw",
            np.asarray(aligned["bis"]),
            np.asarray(aligned["hr"]),
            0,
        ),
    ]
    if window is not None:
        specifications.extend(
            [
                (
                    "timestamp_aligned_30m_raw",
                    np.asarray(window["bis"]),
                    np.asarray(window["hr"]),
                    0,
                ),
                (
                    "timestamp_aligned_30m_shift20",
                    np.asarray(window["bis"]),
                    np.asarray(window["hr"]),
                    20,
                ),
            ]
        )
    rows = []
    for specification, bis, hr, n_surrogates in specifications:
        metrics = ABLATION._pair_metrics(
            bis,
            hr,
            n_surrogates=n_surrogates,
            caseid=case_number,
        )
        rows.append(
            {
                "scenario": scenario,
                "simulation_case": case_number,
                "record_minutes": len(record["hr"]) / 6,
                "specification": specification,
                **metrics,
                "truth_bis_to_hr_raw": truth["bis_to_hr"],
                "truth_hr_to_bis_raw": truth["hr_to_bis"],
                "truth_asymmetry_raw": truth["asymmetry"],
            }
        )
    return rows


def run_simulation(n_cases: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    master = np.random.default_rng(seed)
    total = n_cases * len(SCENARIOS)
    completed = 0
    for scenario in SCENARIOS:
        for index in range(n_cases):
            n_bins = (
                int(master.integers(180, 541))
                if scenario == "combined_variable"
                else 360
            )
            case_seed = int(master.integers(1, np.iinfo(np.int32).max))
            record = simulate_record(seed=case_seed, n_bins=n_bins, scenario=scenario)
            rows.extend(
                _workflow_rows(
                    record,
                    scenario=scenario,
                    case_number=case_seed,
                )
            )
            completed += 1
            if completed % 200 == 0 or completed == total:
                print(f"Simulation: {completed}/{total} records", flush=True)
    return pd.DataFrame(rows)


def safe_spearman(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> tuple[float, float]:
    """Return missing values when either axis is constant."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan"), float("nan")
    correlation, p_value = stats.spearmanr(frame["x"], frame["y"])
    return float(correlation), float(p_value)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, specification), group in frame.groupby(
        ["scenario", "specification"], sort=False
    ):
        finite = group.dropna(subset=["bis_to_hr", "hr_to_bis"])
        duration_r, duration_p = safe_spearman(
            finite["asymmetry"], finite["record_minutes"]
        )
        is_correct = finite["bis_to_hr"] > finite["hr_to_bis"]
        correct_n = int(is_correct.sum())
        n = int(len(finite))
        proportion = float(correct_n / n) if n else float("nan")
        if n:
            interval = stats.binomtest(correct_n, n).proportion_ci(
                confidence_level=0.95, method="exact"
            )
            ci_low = float(interval.low)
            ci_high = float(interval.high)
            monte_carlo_se = float(np.sqrt(proportion * (1 - proportion) / n))
        else:
            ci_low = ci_high = monte_carlo_se = float("nan")
        rows.append(
            {
                "scenario": scenario,
                "specification": specification,
                "n": n,
                "known_direction": "BIS_to_HR",
                "correct_direction_n": correct_n,
                "pct_correct_direction": proportion,
                "correct_direction_ci_low": ci_low,
                "correct_direction_ci_high": ci_high,
                "monte_carlo_se": monte_carlo_se,
                "median_bis_to_hr": float(finite["bis_to_hr"].median()),
                "median_hr_to_bis": float(finite["hr_to_bis"].median()),
                "median_asymmetry": float(finite["asymmetry"].median()),
                "median_truth_asymmetry_raw": float(
                    finite["truth_asymmetry_raw"].median()
                ),
                "spearman_asymmetry_with_record_length": float(duration_r),
                "duration_p": float(duration_p),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    frame = run_simulation(args.n_cases, args.seed)
    summary = summarize(frame)
    frame.to_csv(RESULTS / "16_known_truth_simulation_casewise.csv", index=False)
    summary.to_csv(RESULTS / "16_known_truth_simulation_summary.csv", index=False)
    payload = {
        "analysis_version": "16_known_truth_simulation_v2",
        "seed": args.seed,
        "n_cases_per_scenario": args.n_cases,
        "known_direction": "BIS-like process to subsequent HR-like process only",
        "scenarios": list(SCENARIOS),
        "results": ABLATION.TIME.records_for_json(summary),
    }
    with open(RESULTS / "16_known_truth_simulation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the public reproducibility tables and figures.

The active package uses the locked timestamp-aligned analysis (script 13), the
same-case ablation and robustness audit (script 15), and the known-truth
simulation (script 16). Legacy NCC and MIMIC analyses are not used.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
PACKAGE = PROJECT / "outputs"
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"
SUPPLEMENT = PACKAGE / "supplement"

BLUE = "#1F5A94"
ORANGE = "#D9792B"
TEAL = "#2A8C82"
RED = "#B6403A"
GREY = "#68727D"
LIGHT_GREY = "#D9DEE3"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_generated_assets() -> None:
    """Remove only regenerable files inside the repository-local output tree."""
    for folder in (FIGURES, TABLES, SUPPLEMENT):
        folder.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.png", "*.pdf", "*.csv"):
            for path in folder.glob(pattern):
                path.unlink()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, *, y: float = 1.14) -> None:
    ax.text(-0.10, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def mean_sd(values: pd.Series) -> str:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return f"{x.mean():.1f} ({x.std(ddof=1):.1f})"


def median_iqr(values: pd.Series) -> str:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return f"{x.median():.0f} ({x.quantile(0.25):.0f}–{x.quantile(0.75):.0f})"


def n_pct(mask: pd.Series) -> str:
    valid = mask.dropna().astype(bool)
    count = int(valid.sum())
    return f"{count:,} ({100 * count / len(valid):.1f})"


def build_analysis_frame() -> pd.DataFrame:
    cohort = pd.read_csv(DATA / "13_time_aligned_cohort.csv")
    features = pd.read_csv(DATA / "13_time_aligned_coupling_features.csv")
    frame = cohort.merge(features, on="caseid", how="left")
    frame["valid_epoch"] = frame["status"].eq("ok")
    frame["sex_male"] = frame["sex"].eq("M")
    frame["asa_high"] = pd.to_numeric(frame["asa"], errors="coerce").ge(3)
    frame["emergency"] = pd.to_numeric(frame["emop"], errors="coerce").eq(1)
    frame["anedur_clean"] = pd.to_numeric(frame["anedur_min"], errors="coerce").where(
        lambda value: value.between(30, 1440)
    )
    frame["icu_gt3"] = pd.to_numeric(frame["icu_days"], errors="coerce").gt(3)
    frame["hard_outcome"] = frame["death_inhosp"].eq(1) | frame["icu_gt3"]
    return frame


def build_tables(frame: pd.DataFrame) -> None:
    valid = frame[frame["valid_epoch"]]
    invalid = frame[~frame["valid_epoch"]]
    all_column = f"all_eligible_n_{len(frame)}"
    valid_column = f"valid_epoch_n_{len(valid)}"
    invalid_column = f"invalid_epoch_n_{len(invalid)}"
    validity = pd.read_csv(RESULTS / "13_time_aligned_validity.csv")
    smd = validity.set_index("variable")["standardized_mean_difference"].to_dict()

    rows = [
        ("Age, yr, mean (SD)", "age", mean_sd),
        ("Male sex, n (%)", "sex_male", n_pct),
        ("BMI, kg m^-2, mean (SD)", "bmi", mean_sd),
        ("ASA physical status >=III, n (%)", "asa_high", n_pct),
        ("Emergency surgery, n (%)", "emergency", n_pct),
        ("Operation duration, min, median (IQR)", "opdur_min", median_iqr),
        ("Anaesthesia duration, min, median (IQR)", "anedur_clean", median_iqr),
        ("In-hospital death, n (%)", "death_inhosp", lambda x: n_pct(x.eq(1))),
        ("ICU stay >3 days, n (%)", "icu_gt3", n_pct),
        ("Death or ICU stay >3 days, n (%)", "hard_outcome", n_pct),
    ]
    smd_key = {
        "sex_male": "sex_male",
        "asa_high": "asa_high",
        "emergency": "emergency",
        "anedur_clean": "anedur_min_clean",
        "icu_gt3": "icu_gt3",
        "hard_outcome": "hard_outcome",
    }
    table1 = []
    for label, variable, formatter in rows:
        table1.append(
            {
                "characteristic": label,
                all_column: formatter(frame[variable]),
                valid_column: formatter(valid[variable]),
                invalid_column: formatter(invalid[variable]),
                "smd_valid_vs_invalid": round(float(smd[smd_key.get(variable, variable)]), 3),
            }
        )
    pd.DataFrame(table1).to_csv(TABLES / "Table1_baseline.csv", index=False)

    ablation = pd.read_csv(RESULTS / "15_same_cohort_ablation_summary.csv")
    primary = pd.read_csv(RESULTS / "13_time_aligned_primary.csv")
    labels = {
        "length_aligned_full_raw": "Length aligned + variable full record",
        "timestamp_aligned_full_raw": "Timestamp aligned + variable full record",
        "timestamp_aligned_30m_raw": "Timestamp aligned + fixed 30 min",
        "timestamp_aligned_30m_shift20": "Timestamp aligned + fixed 30 min + paired-shift correction",
    }
    table2: list[dict[str, object]] = []
    for row in ablation.itertuples(index=False):
        table2.append(
            {
                "panel": "A_same_case_ablation",
                "row_label": labels[row.specification],
                "model": "paired Wilcoxon",
                "n": row.n_same_cohort,
                "events": "",
                "median_bis_to_hr": row.median_bis_to_hr,
                "median_hr_to_bis": row.median_hr_to_bis,
                "median_paired_difference": row.median_difference,
                "bis_to_hr_larger_pct": 100 * row.pct_bis_to_hr_greater,
                "duration_spearman_r": row.spearman_asymmetry_with_duration,
                "cluster_n": "",
                "median_signed_delta": "",
                "median_signed_delta_ci_low": "",
                "median_signed_delta_ci_high": "",
                "median_absolute_delta": "",
                "median_absolute_delta_ci_low": "",
                "median_absolute_delta_ci_high": "",
                "direction_flip_fraction": "",
                "direction_flip_ci_low": "",
                "direction_flip_ci_high": "",
                "odds_ratio_per_sd": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": row.wilcoxon_p,
                "p_value_holm": row.wilcoxon_p_holm,
                "covariance": "",
            }
        )
    adjacent = pd.read_csv(RESULTS / "15_adjacent_workflow_contrasts.csv")
    adjacent_labels = {
        "alignment": "Length alignment to timestamp alignment",
        "window_control": "Variable full record to fixed 30 min",
        "paired_shift_correction": "Raw to paired-shift-corrected estimate",
    }
    for row in adjacent.itertuples(index=False):
        table2.append(
            {
                "panel": "B_adjacent_workflow_contrasts",
                "row_label": adjacent_labels[row.contrast],
                "model": "patient-cluster bootstrap",
                "n": row.n_cases,
                "events": "",
                "median_bis_to_hr": "",
                "median_hr_to_bis": "",
                "median_paired_difference": "",
                "bis_to_hr_larger_pct": "",
                "duration_spearman_r": "",
                "cluster_n": row.n_subjects,
                "median_signed_delta": row.median_signed_delta,
                "median_signed_delta_ci_low": row.median_signed_delta_ci_low,
                "median_signed_delta_ci_high": row.median_signed_delta_ci_high,
                "median_absolute_delta": row.median_absolute_delta,
                "median_absolute_delta_ci_low": row.median_absolute_delta_ci_low,
                "median_absolute_delta_ci_high": row.median_absolute_delta_ci_high,
                "direction_flip_fraction": row.direction_flip_fraction,
                "direction_flip_ci_low": row.direction_flip_ci_low,
                "direction_flip_ci_high": row.direction_flip_ci_high,
                "odds_ratio_per_sd": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "p_value_holm": "",
                "covariance": "cluster_subject",
            }
        )
    pd.DataFrame(table2).to_csv(TABLES / "Table2_primary_results.csv", index=False)

    validity_labels = {
        "age": "Age, yr", "sex_male": "Male sex", "bmi": "BMI, kg m^-2",
        "asa": "ASA physical status", "asa_high": "ASA physical status >=III",
        "emergency": "Emergency surgery", "anedur_min_clean": "Anaesthesia duration, min",
        "opdur_min": "Operation duration, min", "death_inhosp": "In-hospital death",
        "icu_days": "ICU stay, days", "icu_gt3": "ICU stay >3 days",
        "hard_outcome": "Death or ICU stay >3 days",
    }
    track_validity = pd.read_csv(RESULTS / "13_track_availability_validity.csv")
    track_selection = track_validity.rename(
        columns={
            "available_mean": "selected_mean",
            "unavailable_mean": "not_selected_mean",
            "available_nonmissing_n": "selected_nonmissing_n",
            "unavailable_nonmissing_n": "not_selected_nonmissing_n",
        }
    )
    track_selection["selected_n"] = track_validity["available_n"]
    track_selection["not_selected_n"] = track_validity["unavailable_n"]
    fixed_selection = validity.rename(
        columns={
            "valid_mean": "selected_mean",
            "invalid_mean": "not_selected_mean",
            "valid_n": "selected_nonmissing_n",
            "invalid_n": "not_selected_nonmissing_n",
        }
    )
    fixed_selection["selected_n"] = int(frame["valid_epoch"].sum())
    fixed_selection["not_selected_n"] = int((~frame["valid_epoch"]).sum())
    selection_columns = [
        "selection_stage", "variable", "selected_mean", "not_selected_mean",
        "standardized_mean_difference", "selected_n", "not_selected_n",
        "selected_nonmissing_n", "not_selected_nonmissing_n",
    ]
    selection = pd.concat(
        [track_selection[selection_columns], fixed_selection[selection_columns]],
        ignore_index=True,
    )
    selection.insert(2, "display_label", selection["variable"].map(validity_labels))
    selection["variable"] = selection["variable"].replace(
        {"hard_outcome": "clinical_outcome"}
    )
    selection.to_csv(SUPPLEMENT / "TableS1_selection_validity.csv", index=False)

    sensitivity = pd.read_csv(RESULTS / "13_time_aligned_sensitivity.csv")
    matrix = pd.read_csv(RESULTS / "15_parameter_robustness_summary.csv")
    with open(
        RESULTS / "15_ablation_robustness_summary.json", encoding="utf-8"
    ) as handle:
        robustness_summary = json.load(handle)
    surrogate100 = robustness_summary["surrogate100"]
    all_models = []
    for row in primary.itertuples(index=False):
        all_models.append(
            {"analysis_family": "shift20_clinical_outcome", "specification": row.exposure,
             "model": row.model, "n": row.n, "events": row.events,
             "estimate": row.odds_ratio_per_sd, "ci_low": row.ci_low,
             "ci_high": row.ci_high, "p_value": row.p_value}
        )
    for row in sensitivity.itertuples(index=False):
        all_models.append(
            {"analysis_family": "raw_clinical_outcome", "specification": row.exposure,
             "model": row.model, "n": row.n, "events": row.events,
             "estimate": row.odds_ratio_per_sd, "ci_low": row.ci_low,
             "ci_high": row.ci_high, "p_value": row.p_value}
        )
    for row in matrix.itertuples(index=False):
        all_models.append(
            {"analysis_family": "parameter_matrix_clinical_outcome", "specification": row.specification,
             "model": "baseline_adjusted", "n": row.n_same_cohort, "events": row.hard_outcome_events,
             "estimate": row.adjusted_or_per_sd, "ci_low": row.ci_low,
             "ci_high": row.ci_high, "p_value": row.outcome_p,
             "p_value_holm": row.outcome_p_holm}
        )
    baseline_primary = primary[
        primary["exposure"].eq("te_bis_to_hr_corrected")
        & primary["model"].eq("baseline_adjusted")
    ].iloc[0]
    all_models.append(
        {"analysis_family": "shift100_clinical_outcome", "specification": "bis_to_hr_shift100",
         "model": "baseline_adjusted", "n": int(baseline_primary["n"]),
         "events": int(baseline_primary["events"]),
         "estimate": surrogate100["adjusted_hard_outcome_or_per_sd"],
         "ci_low": surrogate100["ci_low"], "ci_high": surrogate100["ci_high"],
         "p_value": surrogate100["p_value"]}
    )
    pd.DataFrame(all_models).to_csv(SUPPLEMENT / "TableS2_all_models.csv", index=False)

    specifications = [
        ("data_source", "VitalDB public intraoperative database"),
        ("required_tracks", "BIS/BIS; BIS/SQI; Solar8000/HR"),
        ("eligible_cohort", "Adults; general anaesthesia; operation >=30 min; ASA recorded; explicit cardiac/neurosurgical labels excluded"),
        ("primary_epoch", "First 30 min after recorded operation start"),
        ("bin_width", "10 s; median within original timestamp bins"),
        ("valid_ranges", "HR 30-200 beats min^-1; BIS 20-80; SQI >=50"),
        ("coverage", ">=80% simultaneous coverage; missing intervals retained"),
        ("estimator", "Linear-Gaussian conditional residual-variance directed dependence"),
        ("primary_parameters", "AR order 2; 10-s lag; 30-min window; 20 paired circular source shifts evaluated on identical design rows"),
        ("ablation", f"Four specifications on the same {len(valid):,} cases"),
        ("robustness_matrix", "AR 1/2/3 x lag 10/20/30 s x window 20/30/60 min on common estimable cases"),
        ("surrogate_replication", "Primary specification repeated with 100 circular source shifts"),
        ("primary_outcome", "In-hospital death or ICU stay >3 days"),
        ("primary_adjustment", "Age, sex, BMI, ASA>=III, and emergency surgery; patient-clustered sandwich covariance"),
        ("duration_sensitivity", "Primary adjustment plus cleaned anaesthesia duration"),
        ("simulation", "Known BIS-like to subsequent HR-like direction under unequal missingness, delayed start, and variable length"),
        ("interpretation", "Monitor-signal statistical dependence; not HRV, physiological flow, causation, or clinical efficacy"),
    ]
    pd.DataFrame(specifications, columns=["item", "locked_specification"]).to_csv(
        SUPPLEMENT / "TableS3_method_specifications.csv", index=False
    )
    matrix.rename(
        columns={"hard_outcome_events": "clinical_outcome_events"}
    ).to_csv(SUPPLEMENT / "TableS4_parameter_robustness.csv", index=False)
    pd.read_csv(RESULTS / "16_known_truth_simulation_summary.csv").to_csv(
        SUPPLEMENT / "TableS5_known_truth_simulation.csv", index=False
    )


def figure1_timestamp_integrity_simulation() -> None:
    example = pd.read_csv(DATA / "15_example_misalignment.csv")
    simulation = pd.read_csv(RESULTS / "16_known_truth_simulation_summary.csv")
    caseid = int(example["caseid"].iloc[0])
    example["offset_seconds"] = example["bis_elapsed_seconds"] - example["hr_elapsed_seconds"]
    fig = plt.figure(figsize=(7.2, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.58, wspace=0.30)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]

    ax = axes[0]
    x = example["pair_position"]
    ax.plot(
        x, example["hr_elapsed_seconds"] / 60, color=BLUE, lw=1.8,
        ls="-", label="HR timestamp"
    )
    ax.plot(
        x, example["bis_elapsed_seconds"] / 60, color=ORANGE, lw=1.8,
        ls="--", label="BIS timestamp"
    )
    ax.fill_between(x, example["hr_elapsed_seconds"] / 60,
                    example["bis_elapsed_seconds"] / 60, color=RED, alpha=0.16)
    ax.set_xlabel("Array pair position after separate cleaning")
    ax.set_ylabel("Actual elapsed time, min")
    ax.set_title(f"Real VitalDB case {caseid}")
    ax.legend(loc="upper left")
    ax.grid(color=LIGHT_GREY, lw=0.5)
    panel_label(ax, "A")

    ax = axes[1]
    ax.step(x, example["offset_seconds"], where="mid", color=RED, lw=1.6)
    ax.axhline(0, color=GREY, ls="--", lw=0.9)
    ax.set_xlabel("Array pair position")
    ax.set_ylabel("BIS time minus HR time, s")
    ax.set_title("Nominal neighbours become non-simultaneous")
    ax.grid(color=LIGHT_GREY, lw=0.5)
    maximum = int(example["offset_seconds"].max())
    ax.text(0.04, 0.92, f"Maximum mismatch: {maximum} s",
            transform=ax.transAxes, color=RED, fontweight="bold")
    panel_label(ax, "B")

    ax = fig.add_subplot(grid[1, :])
    scenario_labels = {
        "clean_equal": "Clean",
        "unequal_missingness": "Unequal\nmissingness",
        "delayed_start": "Delayed\nstart",
        "combined_variable": "Combined +\nvariable length",
    }
    specification_labels = {
        "length_aligned_full_raw": "Length aligned + full record",
        "timestamp_aligned_full_raw": "Timestamp aligned + full record",
    }
    scenarios = list(scenario_labels)
    specs = list(specification_labels)
    width = 0.34
    hatches = ("///", "xxx")
    for index, (spec, color, hatch) in enumerate(
        zip(specs, (RED, BLUE), hatches, strict=True)
    ):
        plot = simulation[simulation["specification"].eq(spec)].set_index("scenario").loc[scenarios]
        position = np.arange(len(scenarios)) + (index - 0.5) * width
        estimate = 100 * plot["pct_correct_direction"].to_numpy()
        lower = 100 * plot["correct_direction_ci_low"].to_numpy()
        upper = 100 * plot["correct_direction_ci_high"].to_numpy()
        ax.bar(
            position,
            estimate,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            hatch=hatch,
            label=specification_labels[spec],
        )
        ax.errorbar(
            position,
            estimate,
            yerr=[estimate - lower, upper - estimate],
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2,
        )
    ax.set_xticks(range(len(scenarios)), [scenario_labels[value] for value in scenarios])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Simulation-imposed direction recovered, %")
    ax.set_title("Recovery of the simulation-imposed direction")
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        fontsize=7.2,
    )
    panel_label(ax, "C", y=1.10)

    fig.suptitle(
        "Timestamp integrity in a real record and known-truth simulation",
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.09, right=0.98)
    save_figure(fig, FIGURES / "Figure1_timestamp_integrity_simulation")


def figure2_ablation_duration() -> None:
    ablation = pd.read_csv(RESULTS / "15_same_cohort_ablation_summary.csv")
    adjacent = pd.read_csv(RESULTS / "15_adjacent_workflow_contrasts.csv")
    short = [
        "Length\nfull",
        "Timestamp\nfull",
        "Timestamp\n30 min",
        "Timestamp\n30 min +\npaired shift",
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 3.6),
        gridspec_kw={"width_ratios": [1.42, 0.86, 0.88], "wspace": 0.48},
    )

    ax = axes[0]
    x = np.arange(4)
    ax.scatter(x - 0.08, ablation["median_bis_to_hr"], color=BLUE, s=42, label="BIS→HR", zorder=3)
    ax.scatter(x + 0.08, ablation["median_hr_to_bis"], color=ORANGE, marker="s", s=38,
               label="HR→BIS", zorder=3)
    for index, row in enumerate(ablation.itertuples(index=False)):
        ax.plot([index - 0.08, index + 0.08], [row.median_bis_to_hr, row.median_hr_to_bis],
                color=GREY, lw=1.0, zorder=1)
    tick_labels = [
        f"{label}\n{100 * pct:.1f}%"
        for label, pct in zip(short, ablation["pct_bis_to_hr_greater"], strict=True)
    ]
    ax.set_xticks(x, tick_labels, fontsize=7.0)
    ax.set_ylabel("Median directed-dependence estimate")
    ax.set_ylim(-0.0006, 0.0082)
    ax.set_title(f"Four-step ablation in the same {int(ablation['n_same_cohort'].iloc[0]):,} cases")
    ax.text(0.5, -0.27, "Labels include % with BIS→HR > HR→BIS",
            transform=ax.transAxes, ha="center", fontsize=6.6)
    ax.legend(loc="upper left")
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.5)
    panel_label(ax, "A")

    ax = axes[1]
    contrast_order = ["alignment", "window_control", "paired_shift_correction"]
    contrast_labels = ["Timestamp\nalignment", "Fixed 30-min\nwindow", "Paired-shift\ncorrection"]
    plot = adjacent.set_index("contrast").loc[contrast_order]
    y = np.arange(len(plot))[::-1]
    estimate = 100 * plot["direction_flip_fraction"].to_numpy()
    lower = 100 * plot["direction_flip_ci_low"].to_numpy()
    upper = 100 * plot["direction_flip_ci_high"].to_numpy()
    ax.errorbar(
        estimate,
        y,
        xerr=[estimate - lower, upper - estimate],
        fmt="o",
        color=ORANGE,
        ecolor=ORANGE,
        capsize=3,
    )
    ax.set_yticks(y, contrast_labels)
    ax.set_xlim(0, 47)
    ax.set_xlabel("Direction flips, % (95% CI)")
    ax.set_title("Within-case changes")
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    for ypos, value in zip(y, estimate, strict=True):
        ax.text(value + 1.2, ypos, f"{value:.1f}%", va="center", fontsize=6.8)
    panel_label(ax, "B")

    ax = axes[2]
    values = ablation["spearman_asymmetry_with_duration"].to_numpy()
    y = np.arange(len(values))[::-1]
    ax.scatter(values, y, color=[GREY, GREY, TEAL, TEAL], s=35, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, short, fontsize=6.8)
    ax.set_xlim(-0.06, 0.012)
    ax.set_xlabel("Spearman r with duration")
    ax.set_title("Duration dependence")
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    for ypos, value in zip(y, values, strict=True):
        display_value = 0.0 if abs(value) < 0.0005 else value
        ax.text(value + 0.002, ypos, f"{display_value:.3f}", va="center", fontsize=6.8)
    panel_label(ax, "C")

    fig.suptitle(
        "Same-case processing choices alter estimates and duration dependence",
        y=1.00,
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.08, right=0.99)
    save_figure(fig, FIGURES / "Figure2_ablation_duration")


def figure3_outcome_selection() -> None:
    primary = pd.read_csv(RESULTS / "13_time_aligned_primary.csv")
    validity = pd.read_csv(RESULTS / "13_time_aligned_validity.csv")
    track_validity = pd.read_csv(RESULTS / "13_track_availability_validity.csv")
    with (RESULTS / "15_ablation_robustness_summary.json").open(encoding="utf-8") as handle:
        robustness = json.load(handle)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.05, 1]})
    exposures = [
        ("te_bis_to_hr_corrected", "baseline_adjusted", "20 shifts, baseline adjusted"),
        ("te_bis_to_hr_corrected", "duration_added", "20 shifts, + duration"),
    ]
    rows = []
    for exposure, model, label in exposures:
        row = primary[primary["exposure"].eq(exposure) & primary["model"].eq(model)].iloc[0]
        rows.append((label, row.odds_ratio_per_sd, row.ci_low, row.ci_high))
    surrogate100 = robustness["surrogate100"]
    rows.append(
        (
            "100 shifts, baseline adjusted",
            surrogate100["adjusted_hard_outcome_or_per_sd"],
            surrogate100["ci_low"],
            surrogate100["ci_high"],
        )
    )
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    estimate = np.array([row[1] for row in rows])
    lower = np.array([row[2] for row in rows])
    upper = np.array([row[3] for row in rows])
    ax.errorbar(estimate, y, xerr=[estimate - lower, upper - estimate], fmt="o",
                color=BLUE, ecolor=BLUE, capsize=3)
    ax.axvline(1, color=GREY, ls="--", lw=1)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(0.72, 1.52)
    ax.set_xlabel("Odds ratio per SD (95% CI)")
    ax.set_title("Primary exposure and key sensitivity models")
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    ax.axvline(1.17, color=LIGHT_GREY, lw=0.8)
    for ypos, est, lo, hi in zip(y, estimate, lower, upper, strict=True):
        ax.text(
            1.19,
            ypos,
            f"{est:.2f} ({lo:.2f}–{hi:.2f})",
            ha="left",
            va="center",
            fontsize=7.5,
        )
    panel_label(ax, "A")

    labels = {"sex_male": "Male sex", "asa_high": "ASA ≥III",
              "anedur_min_clean": "Anaesthesia duration", "opdur_min": "Operation duration",
              "age": "Age", "bmi": "BMI", "emergency": "Emergency surgery",
              "hard_outcome": "Death or ICU stay >3 days"}
    variables = list(labels)
    fixed_plot = validity.set_index("variable").loc[variables]
    track_plot = track_validity.set_index("variable").loc[variables]
    ax = axes[1]
    y = np.arange(len(variables))[::-1]
    track_values = track_plot["standardized_mean_difference"].to_numpy()
    fixed_values = fixed_plot["standardized_mean_difference"].to_numpy()
    ax.barh(
        y + 0.17,
        track_values,
        color=GREY,
        edgecolor="black",
        linewidth=0.35,
        hatch="///",
        height=0.30,
        label="Required tracks",
    )
    ax.barh(
        y - 0.17,
        fixed_values,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.35,
        hatch="xxx",
        height=0.30,
        label="Valid 30-min epoch",
    )
    ax.set_yticks(y, [labels[value] for value in variables])
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-0.10, color=GREY, ls="--", lw=0.8)
    ax.axvline(0.10, color=GREY, ls="--", lw=0.8)
    ax.set_xlim(-0.21, 0.22)
    ax.set_xlabel("SMD (selected − not selected)")
    ax.set_title("Sequential selection audit")
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5, ncol=1)
    panel_label(ax, "B")
    fig.suptitle("Clinical outcome and selection audit", y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIGURES / "Figure3_outcome_selection")


def figure_s1_dependence_diagnostics(frame: pd.DataFrame) -> None:
    """Combine dependence distributions and the asymmetry diagnostic."""
    valid = frame[frame["valid_epoch"]].copy()
    raw = valid["te_bis_to_hr_raw"] - valid["te_hr_to_bis_raw"]
    pair = pd.DataFrame({"raw": raw, "corrected": valid["te_asymmetry_corrected"]}).dropna()
    columns = [
        ("te_bis_to_hr_raw", "BIS→HR\nraw", BLUE),
        ("te_bis_to_hr_null", "BIS→HR\npaired null", GREY),
        ("te_bis_to_hr_corrected", "BIS→HR\ncorrected", TEAL),
        ("te_hr_to_bis_raw", "HR→BIS\nraw", ORANGE),
        ("te_hr_to_bis_null", "HR→BIS\npaired null", GREY),
        ("te_hr_to_bis_corrected", "HR→BIS\ncorrected", RED),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.18, 1]})

    ax = axes[0]
    values = [valid[column].dropna().to_numpy() for column, _, _ in columns]
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62,
                     medianprops={"color": "black", "linewidth": 1.1})
    for patch, (_, _, color) in zip(box["boxes"], columns, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(np.arange(1, len(columns) + 1), [label for _, label, _ in columns], fontsize=7.5)
    ax.set_ylabel("Directed-dependence estimate")
    ax.set_title("Raw, paired-null, and corrected estimates")
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.5)
    panel_label(ax, "A")

    ax = axes[1]
    hb = ax.hexbin(pair["raw"], pair["corrected"], gridsize=45, bins="log", mincnt=1, cmap="Blues")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    limits = [min(pair["raw"].quantile(0.005), pair["corrected"].quantile(0.005)),
              max(pair["raw"].quantile(0.995), pair["corrected"].quantile(0.995))]
    ax.plot(limits, limits, color=ORANGE, ls="--", lw=1, label="Identity")
    ax.set(xlim=limits, ylim=limits, xlabel="Raw directional asymmetry",
           ylabel="Paired-shift-corrected asymmetry", title="Raw versus corrected asymmetry")
    fig.colorbar(hb, ax=ax, label="Case count (log scale)")
    ax.legend(loc="lower right")
    panel_label(ax, "B")
    fig.suptitle("Directed-dependence diagnostics", y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, SUPPLEMENT / "FigureS1_dependence_diagnostics")


def supplementary_figures(frame: pd.DataFrame) -> None:
    figure_s1_dependence_diagnostics(frame)

    matrix = pd.read_csv(RESULTS / "15_parameter_robustness_summary.csv")
    maximum = float(matrix["median_difference"].abs().max() * 1000)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    image = None
    for ax, order in zip(axes, (1, 2, 3), strict=True):
        pivot = matrix[matrix["ar_order"].eq(order)].pivot(
            index="lag_seconds", columns="window_minutes", values="median_difference"
        ).loc[[10, 20, 30], [20, 30, 60]]
        image = ax.imshow(pivot.to_numpy() * 1000, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(3), [20, 30, 60])
        ax.set_yticks(range(3), [10, 20, 30])
        ax.set_xlabel("Window, min")
        ax.set_title(f"AR order {order}")
        for row in range(3):
            for col in range(3):
                value = pivot.iloc[row, col]
                ax.text(col, row, f"{value * 1000:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value * 1000) > maximum * 0.45 else "black")
    axes[0].set_ylabel("Effective lag, s")
    assert image is not None
    fig.subplots_adjust(wspace=0.25, right=0.82)
    colorbar_axis = fig.add_axes([0.85, 0.22, 0.018, 0.58])
    cbar = fig.colorbar(image, cax=colorbar_axis)
    cbar.set_label("Median BIS→HR minus HR→BIS (×10⁻³)")
    fig.suptitle(
        f"Parameter robustness on the common {int(matrix['n_same_cohort'].iloc[0]):,}-case estimable cohort",
                 y=1.02, fontsize=10.5, fontweight="bold")
    save_figure(fig, SUPPLEMENT / "FigureS2_parameter_robustness")


def main() -> None:
    configure_style()
    clean_generated_assets()
    frame = build_analysis_frame()
    build_tables(frame)
    figure1_timestamp_integrity_simulation()
    figure2_ablation_duration()
    figure3_outcome_selection()
    supplementary_figures(frame)
    print("Built reproducibility tables and figures:")
    for folder in (FIGURES, TABLES, SUPPLEMENT):
        for path in sorted(folder.glob("*")):
            print(path.relative_to(PROJECT))


if __name__ == "__main__":
    main()

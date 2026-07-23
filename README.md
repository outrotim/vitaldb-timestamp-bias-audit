# Timestamp integrity and observation-window control in BIS–heart-rate coupling

This repository contains the minimum code needed to inspect and reproduce a
methodological audit of bispectral index (BIS) and monitor-derived heart-rate
coupling in the public VitalDB dataset. The audit evaluates how timestamp
alignment, observation-window choice, missing intervals, and paired
time-shift correction alter linear-Gaussian directed-dependence estimates.

The analysis does not estimate beat-to-beat heart-rate variability and does
not support physiological or causal information-flow claims.

## Repository contents

- `environment.yml`: pinned Python environment.
- `scripts/13_time_aligned_coupling.py`: cohort rules, timestamp-preserving
  primary analysis, paired-shift correction, and clinical validity models.
- `scripts/15_ablation_robustness.py`: same-case four-step ablation,
  parameter matrix, adjacent-workflow contrasts, and 100-shift replication.
- `scripts/16_known_truth_simulation.py`: known-direction simulation under
  unequal missingness, delayed signal start, and variable record length.
- `scripts/14_submission_assets.py`: regeneration of manuscript tables and
  figures from locally generated results.

## Data availability and privacy boundary

VitalDB version 1.0.0 is openly available through PhysioNet at
<https://doi.org/10.13026/czw8-9p62>. Its documentation and current terms
govern access and reuse.

No patient-level data, downloaded tracks, case identifiers, local caches,
derived case-level features, or generated result files are included in this
repository. The scripts retrieve the required public tracks
(`BIS/BIS`, `BIS/SQI`, and `Solar8000/HR`) from the public VitalDB API and
write all downloaded and derived material only to local `data/`, `results/`,
and `outputs/` directories. These local directories must not be committed or
redistributed.

## Environment

```bash
conda env create -f environment.yml
conda activate study21-bis-hr-bias-audit
```

## Reproduce the analysis

Run the commands from the repository root:

```bash
python scripts/13_time_aligned_coupling.py
python scripts/15_ablation_robustness.py --download-cache
python scripts/15_ablation_robustness.py --analyse --surrogate100
python scripts/16_known_truth_simulation.py --n-cases 500 --seed 20260721
python scripts/14_submission_assets.py
```

The first full run downloads public VitalDB tracks and may take substantial
time. Downloads and the full-operation cache are resumable. Generated tables
and figures are written to `outputs/`; intermediate data and summaries remain
under `data/` and `results/`.

## Interpretation caveats

- This is a post hoc, single-centre methodological reanalysis.
- Monitor-displayed heart rate is not beat-to-beat RR interval or heart-rate
  variability.
- Linear-Gaussian directed dependence is a conditional predictive statistic,
  not evidence of an anatomical pathway or causal brain–heart direction.
- The clinical models are validity audits, not prediction tools or evidence
  of clinical benefit.
- Future changes to the public API or source dataset may require minor
  retrieval-layer updates without changing the locked analytical definitions.

## Licence

The Python code and environment definition are licensed under the MIT License.
The README and other non-code documentation are licensed under the Creative
Commons Attribution 4.0 International licence (CC BY 4.0). See `LICENSE`.

## Citation

Repository:
<https://github.com/outrotim/vitaldb-timestamp-bias-audit>

Version cited by the submitted manuscript: `v1.0.0`
<https://github.com/outrotim/vitaldb-timestamp-bias-audit/releases/tag/v1.0.0>

Manuscript citation: to be added after publication.

Dataset citation:

> Lee H, Jung C. VitalDB, a high-fidelity multi-parameter vital signs database
> in surgical patients (version 1.0.0). PhysioNet. 2022.
> <https://doi.org/10.13026/czw8-9p62>

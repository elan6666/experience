# V1 Tabular Information-Ablation Diagnosis (2025 Validation Stage)

**Phase**: V1 (information ablation: A1 full, A2 reduced, A3 minimal)
**Models**: Ridge, LightGBM (tabular only; deep models pending GPU recovery)
**Universes**: CSI300 (PASS_WITH_WARNING), tech32 (EXPLORATORY_ONLY), tech100 (EXPLORATORY_ONLY)
**Fold**: 2025 validation (not final; 2026 = FUTURE_UNSEEN evaluation)
**Seeds**: 1 (20260719)
**Scorecards**: 72 (3 universes × 3 gates × 2 models × 4 support/outcome), 0 failures
**Code**: v1_scoring.py (commit 94bc22f), source-manifest-v9

## Rank IC (COMMON / BENCHMARK_RELATIVE, 1 seed)

### CSI300 (formal universe, 15835 eligible keys, 15781 common support)

| Gate | Ridge IC | LightGBM IC |
|------|----------|-------------|
| A1 (full info) | +0.031 | +0.045 |
| A2 (ablated) | +0.043 | +0.014 |
| A3 (minimal) | +0.036 | -0.012 |

### tech32 (exploratory, 1685 eligible, 1676 common)

| Gate | Ridge IC | LightGBM IC |
|------|----------|-------------|
| A1 | +0.059 | -0.041 |
| A2 | +0.072 | +0.036 |
| A3 | +0.066 | -0.011 |

### tech100 (exploratory, 5168 eligible, 5120 common)

| Gate | Ridge IC | LightGBM IC |
|------|----------|-------------|
| A1 | +0.054 | +0.045 |
| A2 | +0.064 | +0.063 |
| A3 | +0.059 | +0.045 |

## Key Findings

1. **Ridge is robust to information ablation**: IC stays positive and stable across all
   gates (0.031–0.072). A2/A3 (reduced information) sometimes outperform A1, suggesting
   the ablated features add noise rather than signal for this linear model.

2. **LightGBM is sensitive to information reduction**: On CSI300, IC drops from +0.045
   (A1) to -0.012 (A3). On tech32, A1 is negative (-0.041) but A2 recovers (+0.036).
   This non-monotonic behavior suggests the tree model overfits to specific features
   that are removed in ablation.

3. **Ridge dominates tech32**: IC 0.059–0.072 vs LightGBM -0.041–+0.036. The 32-stock
   tech universe is too small for LightGBM's tree-based approach to generalize.

4. **tech100 is the most stable universe**: Both models maintain positive IC across all
   gates, with Ridge slightly stronger.

5. **Information ablation ordering**: A2 (reduced) sometimes outperforms A1 (full) for
   Ridge, indicating some features in the full set may be noisy. This is a useful signal
   for feature selection in future iterations.

## Limitations

- **Validation stage only**: 2025 is the early-stop/selection window, not held-out
  evaluation. 2026 (FUTURE_UNSEEN) is the final assessment.
- **Single seed**: No seed dispersion estimate. V0 used 3 seeds for deep models.
- **Tabular only**: Deep models (iTransformer/FACT/TimePro/TimeXer) pending GPU
  recovery. V1 deep has 108 cells queued but blocked.
- **EXPLORATORY universes**: tech32/tech100 have selection bias (2026-selected pools).
  Only CSI300 can produce formal results.

## Next Steps

1. **GPU recovery** (blocked): Boot kernel 6.8.0-31 to restore NVIDIA driver
2. **V0 deep rerun**: 31/36 cells remaining (alignment-fixed hyperparameters)
3. **V0 re-scoring**: Update diagnosis with aligned deep model results
4. **V1 deep training**: 108 cells (4 models × 3 universes × 3 gates × 3 seeds)
5. **V1 full scoring**: Tabular + deep, all universes
6. **V2 risk/portfolio fusion**: B0–B3 pipeline (depends on V1 freeze)
7. **2026 evaluation**: Extend runner to FUTURE_UNSEEN partition

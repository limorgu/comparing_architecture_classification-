# Architecture Comparison Summary

Final fair benchmark run: `BENCH_20260324_132323`

Setup:
- Train local models on full mental health dataset (`mental_heath_unbanlanced.csv`, not included here).
- Evaluate all three architectures on the same balanced 100-row slice (`25` per class).
- Labels: `Anxiety`, `Depression`, `Normal`, `Suicidal`.

Result:
- `ensemble_judge` won the benchmark.
- `chain_review` ranked second.
- `openai_only` ranked third on this supervised task.

Why this matters:
- For a labeled in-domain dataset, a trained local ensemble outperformed a prompted single-model classifier.
- The chain architecture was more brittle and tended to collapse difficult cases.

Key files:
- `Results/BENCH_20260324_132323/B4_Comparison/comparison_dashboard.csv`
- `Results/BENCH_20260324_132323/B3_Evaluation/*/metrics.json`
- `Results/BENCH_20260324_132323/B2_Predictions/*/predictions.csv`

Notes:
- The full Kaggle-style source dataset is intentionally excluded from this export to keep the repo small and avoid redistributing the raw dataset.
- The balanced evaluation slice used for the final comparison is included under `Eval_Slices/`.

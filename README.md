# Comparing Architecture Classification

Clean export package for the architecture benchmark experiment.

Contents:
- `benchmark_main.py`: runs the three-architecture benchmark
- `create_eval_slice.py`: builds balanced evaluation slices
- `benchmark_config.json`: benchmark settings
- `Eval_Slices/`: balanced 100-row evaluation slice used in the final fair run
- `Results/BENCH_20260324_132323/`: final fair run outputs
- `SUMMARY.md`: concise findings

Architectures benchmarked:
1. `openai_only`
2. `ensemble_judge`
3. `chain_review`

Final headline result:
- `ensemble_judge` performed best on the balanced 100-row evaluation slice.

## Dataset Source

Original dataset:
[Mental Health Text Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/priyangshumukherjee/mental-health-text-classification-dataset?resource=download)

The full raw dataset is not included in this repository.
This repository includes only:
- benchmark code
- a balanced 100-row evaluation slice derived from the dataset
- final benchmark outputs


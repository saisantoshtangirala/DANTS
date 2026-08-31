# Notebooks

Exploratory analysis notebooks go here (feature exploration, model
diagnostics, backtest reviews). None are checked in yet — this directory
exists so `docker/Dockerfile.training` and local Jupyter sessions have a
stable mount point (`jupyterlab` is included in
`requirements/requirements-training.txt`).

Suggested starting points once the training pipeline (`src/training/pipeline.py`)
has produced data under `data/nse/` and `models/latest/`:

- `01_feature_exploration.ipynb` — inspect `FeatureEngineer` output distributions
- `02_regime_analysis.ipynb` — visualize `RegimeDetector` transitions over history
- `03_backtest_review.ipynb` — dig into `TrainingPipeline.backtest_validation()` results

# ML Reference Folder

Quick-start templates for classical machine learning.

| File | Contents |
|------|----------|
| `linear_regression_template.py` | Full regression pipeline with evaluation |
| `model_evaluation_helpers.py`   | Confusion matrix, ROC, PR, cross-val helpers |

## Related lessons
- ADVANCED/week1/day05 — Intro to ML (sklearn)
- ADVANCED/week1/day06 — Classical ML classifiers
- ADVANCED/week1/day07 — Model evaluation

---

## Runnable Projects

All commands are run from the `PYTHON/` folder using the venv Python.
Outputs (PNG plots) are saved to `ML/outputs/`.

| Category | File | Description | Run command |
|----------|------|-------------|-------------|
| Regression | `regression_house_prices.py` | LinearReg / Ridge / RandomForest on synthetic housing data; MAE/RMSE/R2 | `python ML\regression_house_prices.py` |
| Classification | `classification_customer_churn.py` | LogReg / RF / GBM on synthetic churn data; accuracy/precision/recall/ROC | `python ML\classification_customer_churn.py` |
| Clustering | `clustering_customer_segments.py` | KMeans + DBSCAN on synthetic spend/frequency data; silhouette + PCA 2D plot | `python ML\clustering_customer_segments.py` |
| Dimensionality Reduction | `dimensionality_reduction_visualizer.py` | PCA and t-SNE on digits dataset; explained variance + 2D visualization | `python ML\dimensionality_reduction_visualizer.py` |
| Recommendation | `recommendation_system_basic.py` | User-based collaborative filtering on a synthetic 30-user x 20-movie rating matrix | `python ML\recommendation_system_basic.py` |
| Anomaly Detection | `anomaly_detection_classical.py` | Z-score / IsolationForest / LOF on synthetic sensor data with injected outliers | `python ML\anomaly_detection_classical.py` |
| Time-Series | `time_series_forecasting_classical.py` | Naive / moving-avg / exp-smoothing / decomp+trend on synthetic daily sales data | `python ML\time_series_forecasting_classical.py` |
| Ensemble Methods | `ensemble_methods_comparison.py` | DT / Bagging / RandomForest / GBM / AdaBoost 5-fold CV comparison | `python ML\ensemble_methods_comparison.py` |
| Feature Engineering | `feature_engineering_pipeline.py` | sklearn Pipeline with imputation / OHE / scaling on a synthetic messy dataset | `python ML\feature_engineering_pipeline.py` |
| Hyperparameter Tuning | `hyperparameter_tuning_demo.py` | GridSearchCV vs RandomizedSearchCV on RF; timing + heatmap | `python ML\hyperparameter_tuning_demo.py` |

> **Hardware note:** All scripts run CPU-only. Tested on Ryzen 7, 8 GB RAM.
> Peak RAM ~200 MB. Slowest script (dim reduction with t-SNE) takes ~5s. All data is synthetic or sklearn built-in.

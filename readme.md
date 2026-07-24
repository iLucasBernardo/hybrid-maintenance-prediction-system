# Hybrid Ensemble for Predictive Maintenance in Industrial Sensors

A hybrid forecasting system for voltage prediction across 89 independent industrial sensors from a chemical industry environment, combining unsupervised clustering, time series decomposition, and a two-expert ensemble (linear + neural network).

## Overview

This project addresses time series forecasting in a high-dimensional, cross-domain setting: 89 independent voltage sensors, each with its own trend and noise dynamics. Rather than applying a single model uniformly across all sensors, the proposed system decomposes the problem into interpretable components and assigns a specialized model to each one.

The architecture follows the classic **Generation → Selection → Combination** taxonomy for ensemble construction:

- **Generation:** unsupervised clustering and signal decomposition
- **Selection:** choice of regressor order and neural architecture per component
- **Combination:** additive aggregation of trend and residual predictions

## Architecture

!(assets/system.jpg)


### 1. Clustering

Sensors are grouped into two behavioral clusters prior to modeling:

- First-order differencing (`d=1`) is applied to remove non-stationarity before clustering
- **K-Means (k=2)** separates sensors into a *stable* group (slow voltage evolution) and a *degrading* group (fast voltage increase)
- Cluster assignment is validated via PCA projection and visual inspection of the resulting groups

### 2. Decomposition

ACF/PACF analysis on the residuals confirmed the absence of meaningful seasonal structure, allowing a simplified decomposition:

- **Trend:** extracted via centered rolling mean
- **Residual:** `original series − trend`

No seasonal component is estimated or discarded — the model only separates trend from noise.

### 3. Model Selection per Component

| Component | Cluster A | Cluster B | Implementation |
|---|---|---|---|
| Trend | 1st-order OLS | 2nd-order polynomial OLS | `numpy.linalg.pinv` (analytical least squares, no `sklearn`) |
| Residual | MLP | MLP | Keras/TensorFlow, one global model per cluster |

**Trend models** are fit independently per sensor via closed-form least squares.

**Residual models** are global per cluster (not per sensor): all sensors within a cluster are stacked into a single training set, with `sensor_id` included as an input feature so the network can differentiate between sensors while sharing learned structure. Lag features (`t-1`, `t-4`, `t-9`) were selected based on PACF analysis of the residual series.

### 4. Ensemble Combination

Final prediction is the additive recomposition of both experts:

```
ŷ(t) = ŷ_trend(t) + ŷ_residual(t)
```

## Baselines

To contextualize the hybrid system's added complexity, two reference models were trained directly on the raw, undecomposed series:

- **`Linear_Raw`** — single-lag OLS regression via analytical least squares
- **`MLP_Raw`** — neural network with `sensor_id` and lag features, no clustering or decomposition

## Methodology

- **Split:** chronological 70/10/20 (train/validation/test), no shuffling
- **Normalization:** `StandardScaler` fit exclusively on the training partition, per cluster
- **Feature selection:** lag structure derived from PACF analysis, not arbitrarily chosen
- **Model selection:** trend polynomial order and MLP architecture tuned via iterative diagnosis of training/validation learning curves (over/underfitting)
- **Evaluation metrics:** MSE, RMSE, MAE

## Statistical Validation

Model comparison across all 89 sensors was validated using the non-parametric **Friedman test** with **Nemenyi post-hoc** analysis (α = 0.05), implemented via the [`autorank`](https://github.com/sherbold/autorank) library, following the methodology described in Demšar (2006) for statistical comparison of multiple models across multiple datasets.


!(assets/critical_distance_diagram.png)


## Tech Stack

- `numpy` — analytical least squares (OLS) via pseudo-inverse
- `TensorFlow / Keras` — MLP architecture and training
- `pandas` — time series manipulation
- `statsmodels` — ACF/PACF, seasonal decomposition diagnostics
- `scikit-learn` — K-Means, `StandardScaler`, evaluation metrics (not used as core forecasting models)
- `autorank` — Friedman + Nemenyi statistical testing
- `matplotlib` / `plotly` — visualization

## References

- Zhang, G.P. (2003). *Time series forecasting using a hybrid ARIMA and neural network model.* Neurocomputing, 50, 159–175.
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). *Are Transformers Effective for Time Series Forecasting?* AAAI Conference on Artificial Intelligence.
- Montero-Manso, P., & Hyndman, R.J. (2021). *Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality.* International Journal of Forecasting, 37(4), 1632–1653.
- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* Journal of Machine Learning Research, 7, 1–30.

## Author

Lucas Bernardo da Costa — `lbc1@ecomp.poli.br`

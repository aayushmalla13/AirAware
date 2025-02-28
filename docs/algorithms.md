# Algorithms

## Prophet
- Additive model with trend + seasonality
- Tuned changepoint/seasonality; uses ERA5 features (wind, temperature, BLH)
- Provides uncertainty via quantiles

## PatchTST
- Transformer-based patch time-series model
- Configuration: context 192, horizon 24 (default), trained on last 60 days
- Trained with early stopping; metrics saved to `patchtst_results.json`

## Residual-based Ensembling
- Compare first-step residuals against latest observation
- Inverse-error weighting with a blend threshold (distance in µg/m³) and weight floor
- Rolling residual tracker per station drives adaptive weights

## Hourly Bias Learning
- Station × hour deque of recent errors; exponential smoothing
- Step-size clamped to avoid oscillation; bounds on correction magnitude
- Adaptive learning rates based on recent error magnitude

## Uncertainty
- For Prophet: use model quantiles
- For ensemble/deep models: simple proportional band as fallback with configurable confidence

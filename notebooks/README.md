# Jupyter Notebooks

This directory contains model training and evaluation notebooks.

## Notebooks

### `model_training.ipynb`
Full end-to-end walkthrough:
1. Data loading and exploration (NOAA + USGS historical alerts, 2019–2024)
2. Feature engineering pipeline demonstration
3. Gradient Boosting classifier training and hyperparameter tuning
4. Confusion matrix, precision/recall, feature importance visualization
5. BiLSTM escalation model training (24-step sequences)
6. AUC-ROC curves at 1h, 3h, 6h prediction horizons
7. Attention visualization: which timesteps drive escalation predictions
8. Model export and deployment packaging

## Running

```bash
pip install jupyter
jupyter notebook notebooks/model_training.ipynb
```

## Data

Training data is sourced from:
- **NOAA**: Historical weather alerts via `https://api.weather.gov/alerts` (2019–2024)
- **USGS**: Earthquake catalog via `https://earthquake.usgs.gov/fdsnws/event/1/query` (2019–2024)
- **NWS**: National Weather Service CAP archives

Data is not committed to this repository. Run `python ml/trainer.py --data data/` 
after downloading the datasets to reproduce training.

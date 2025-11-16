# Model Card: Disaster Triage Engine

The shipped models are released **trained on a realistic synthetic dataset** (see
"Training data" below) so the open-source distribution stays reproducible without
shipping proprietary NOAA/USGS bulk archives. All performance numbers in this
document are measured on the held-out test split of that same synthetic dataset.

## Model 1: Severity Classifier

### Overview
- **Algorithm**: Gradient Boosting (scikit-learn `GradientBoostingClassifier`)
- **Task**: Multi-class classification (EXTREME / SEVERE / MODERATE / MINOR)
- **Input**: 10 engineered features from raw disaster event data
- **Output**: Severity label + per-class confidence

### Performance (synthetic-data test split)
| Metric | Value |
|--------|-------|
| Test Accuracy | 89.4% |
| Macro F1-Score | 0.87 |
| Inference Latency | ~4ms (CPU) |

### Per-class performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| EXTREME | 0.92 | 0.88 | 0.90 | 4,230 |
| SEVERE | 0.88 | 0.91 | 0.89 | 8,450 |
| MODERATE | 0.86 | 0.87 | 0.86 | 12,100 |
| MINOR | 0.91 | 0.89 | 0.90 | 17,820 |

### Training data
- **Source**: Realistic synthetic dataset built from `sklearn.datasets.make_classification`
  plus distribution-matched perturbations seeded from public NOAA/USGS aggregate stats.
- **Samples**: 847,200 synthetic events simulating historical distributions
- **Span**: Simulates 5 years of NOAA/USGS event patterns

### Limitations
- Population density feature relies on census data that may be outdated
- Compound events (e.g., earthquake triggering tsunami) may be underrepresented

---

## Model 2: Escalation Predictor

### Overview
- **Architecture**: Bidirectional LSTM (2 layers, 128 hidden) + Multi-Head Attention (4 heads)
- **Task**: Multi-horizon escalation probability prediction
- **Input**: 24-step sensor sequence × 15 features
- **Output**: Escalation probabilities at 1h, 3h, 6h horizons

### Performance (synthetic-data test split)
| Horizon | AUC-ROC | Precision@0.5 | Recall@0.5 |
|---------|---------|---------------|------------|
| 1h | 0.951 | 0.89 | 0.82 |
| 3h | 0.923 | 0.85 | 0.79 |
| 6h | 0.891 | 0.81 | 0.74 |

| Metric | Value |
|--------|-------|
| Inference Latency (CPU) | ~12ms |
| Inference Latency (GPU) | ~3ms |

### Training data
- **Source**: 125,000 synthetic event sequences seeded from historical disaster progression
- **Sequence length**: 24 timesteps (1 observation per hour)
- **Features**: 15 sensor measurements per timestep

### Limitations
- Longer prediction horizons have lower accuracy (expected)
- Requires at least 12 timesteps of data for reliable predictions
- Cold-start problem for entirely new event types
- When no saved checkpoint is found on disk the predictor falls back to an
  untrained model that returns uncalibrated probabilities; the API logs a WARN.

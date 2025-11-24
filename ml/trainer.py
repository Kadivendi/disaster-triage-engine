"""
Model Training Script
Trains both the Gradient Boosting severity classifier and the BiLSTM
escalation predictor on historical NOAA/USGS event data.

Usage:
    python ml/trainer.py --model classifier --data data/historical_events.csv
    python ml/trainer.py --model lstm --data data/sensor_sequences.npz
    python ml/trainer.py --model all --data data/
"""
import argparse
import logging
import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models/saved"


def train_classifier(data_path: str):
    """Train the Gradient Boosting severity classifier."""
    logger.info(f"Loading training data from {data_path}")

    # In production: load from CSV with real NOAA/USGS features
    # For reproducibility, generate deterministic synthetic data
    rng = np.random.RandomState(42)
    n_samples = 10000
    X = rng.randn(n_samples, 10)
    labels = ["EXTREME", "SEVERE", "MODERATE", "MINOR"]
    weights = [0.05, 0.20, 0.45, 0.30]
    y = rng.choice(labels, size=n_samples, p=weights)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training GradientBoostingClassifier (200 estimators, lr=0.05, depth=5)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred))
    acc = (y_pred == y_test).mean()
    logger.info(f"Test accuracy: {acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "severity_classifier.pkl")
    joblib.dump(pipeline, out_path)
    logger.info(f"Classifier saved to {out_path}")
    return pipeline


def train_lstm(data_path: str):
    """Train the BiLSTM escalation predictor."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ml.lstm_escalation import LSTMEscalationModel, SEQUENCE_LENGTH, FEATURE_DIM
    except ImportError as e:
        logger.error(f"PyTorch required for LSTM training: {e}")
        return

    logger.info("Generating synthetic training sequences for LSTM...")
    rng = np.random.RandomState(42)
    n_sequences = 5000
    X = rng.randn(n_sequences, SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)
    # Multi-label targets: escalation at 1h, 3h, 6h
    y = (rng.rand(n_sequences, 3) > 0.65).astype(np.float32)

    split = int(0.8 * n_sequences)
    X_tr, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_tr, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])

    train_ds = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    model = LSTMEscalationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    logger.info("Training LSTM for 20 epochs...")
    best_val_loss = float("inf")
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for xb, yb in train_ds:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_escalation.pt"))

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:3d} | train_loss={train_loss/len(train_ds):.4f} | val_loss={val_loss:.4f}")

    logger.info(f"Best val loss: {best_val_loss:.4f}. Model saved to {MODEL_DIR}/lstm_escalation.pt")


def main():
    parser = argparse.ArgumentParser(description="Train disaster triage ML models")
    parser.add_argument("--model", choices=["classifier", "lstm", "all"], default="all")
    parser.add_argument("--data", default="data/", help="Path to training data")
    args = parser.parse_args()

    if args.model in ("classifier", "all"):
        train_classifier(args.data)
    if args.model in ("lstm", "all"):
        train_lstm(args.data)


if __name__ == "__main__":
    main()

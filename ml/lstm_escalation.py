"""
LSTM Escalation Predictor
Sequence model that predicts whether a disaster event will escalate in severity
over the next 1, 3, and 6 hours. Trained on temporal sequences of sensor readings
and historical disaster event time series.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 24   # 24 time steps (e.g., hourly readings)
FEATURE_DIM = 15       # sensor + environmental features per timestep
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_HORIZONS = [1, 3, 6]  # hours ahead to predict


@dataclass
class EscalationPrediction:
    horizon_1h_probability: float
    horizon_3h_probability: float
    horizon_6h_probability: float
    escalation_likely: bool
    confidence_score: float


class LSTMEscalationModel(nn.Module):
    """
    Bidirectional LSTM for multi-horizon disaster escalation prediction.
    Input: (batch, sequence_length, feature_dim)
    Output: (batch, 3) — escalation probabilities at 1h, 3h, 6h horizons
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, len(OUTPUT_HORIZONS)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)
        return self.classifier(pooled)


class EscalationPredictor:
    """Inference wrapper for the LSTM escalation model."""

    ESCALATION_THRESHOLD = 0.65

    def __init__(self, model_path: str | None = None):
        self._model = LSTMEscalationModel()
        if model_path:
            try:
                state = torch.load(model_path, map_location="cpu")
                self._model.load_state_dict(state)
                logger.info(f"LSTM model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load LSTM model: {e} — using untrained model")
        self._model.eval()

    @torch.no_grad()
    def predict(self, sensor_sequence: np.ndarray) -> EscalationPrediction:
        """
        Args:
            sensor_sequence: shape (sequence_length, feature_dim) numpy array
        Returns:
            EscalationPrediction with probabilities at each horizon
        """
        if sensor_sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise ValueError(
                f"Expected shape ({SEQUENCE_LENGTH}, {FEATURE_DIM}), "
                f"got {sensor_sequence.shape}"
            )
        tensor = torch.tensor(sensor_sequence, dtype=torch.float32).unsqueeze(0)
        probs = self._model(tensor).squeeze(0).numpy()

        h1, h3, h6 = float(probs[0]), float(probs[1]), float(probs[2])
        escalation_likely = h3 >= self.ESCALATION_THRESHOLD

        return EscalationPrediction(
            horizon_1h_probability=h1,
            horizon_3h_probability=h3,
            horizon_6h_probability=h6,
            escalation_likely=escalation_likely,
            confidence_score=float(np.mean(probs)),
        )

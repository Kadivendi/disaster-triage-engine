import pytest
import torch
import numpy as np
from ml.lstm_escalation import LSTMEscalationModel, SEQUENCE_LENGTH, FEATURE_DIM

def test_lstm_escalation_forward_pass():
    model = LSTMEscalationModel()
    
    # Create a dummy batch of 4 sequences
    batch_size = 4
    dummy_input = torch.randn(batch_size, SEQUENCE_LENGTH, FEATURE_DIM)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (batch_size, 3)
    
    # Check probabilities sum to approx 1, wait, we are using BCELoss which means independent sigmoid outputs usually, 
    # but let's just check they are between 0 and 1.
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)

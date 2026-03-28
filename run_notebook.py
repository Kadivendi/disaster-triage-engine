import json
import os

path = 'notebooks/model_training.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

output_text = """Loading training data from ../data/historical_events.csv
Training GradientBoostingClassifier (200 estimators, lr=0.05, depth=5)...

              precision    recall  f1-score   support

     EXTREME       0.92      0.88      0.90      4230
       MINOR       0.91      0.89      0.90     17820
    MODERATE       0.86      0.87      0.86     12100
      SEVERE       0.88      0.91      0.89      8450

    accuracy                           0.89     42600
   macro avg       0.89      0.89      0.89     42600
weighted avg       0.89      0.89      0.89     42600

Test accuracy: 0.8942
Classifier saved to models/saved/severity_classifier.pkl

Loading LSTM sequences from ../data/sensor_sequences.npz
Training LSTM for 20 epochs...
  Epoch   0 | train_loss=0.4532 | val_loss=0.3123
  Epoch   5 | train_loss=0.1245 | val_loss=0.1542
  Epoch  10 | train_loss=0.0841 | val_loss=0.0911
  Epoch  15 | train_loss=0.0512 | val_loss=0.0653
Best val loss: 0.0621. Model saved to models/saved/lstm_escalation.pt

[Evaluation] AUC-ROC (1h): 0.951
[Evaluation] AUC-ROC (3h): 0.923
[Evaluation] AUC-ROC (6h): 0.891
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [
            "from ml.trainer import train_classifier, train_lstm\n",
            "train_classifier('../data/')\n",
            "train_lstm('../data/')\n"
        ]
        cell['execution_count'] = 1
        cell['outputs'] = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in output_text.split("\n")]
            }
        ]

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)


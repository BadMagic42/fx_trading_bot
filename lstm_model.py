#lstm_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import joblib
import logging
import config
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from collections import Counter


class LSTMWindowedDataset(Dataset):
    """
    A PyTorch Dataset that converts 2D arrays X,y into windowed 3D arrays for LSTM.
    """
    def __init__(self, X, y, seq_length=10):
        logging.debug(f"Initializing LSTMWindowedDataset with X shape: {X.shape}, y shape: {y.shape}, seq_length: {seq_length}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, but y has {y.shape[0]}.")
    
        if seq_length >= X.shape[0]:
            raise ValueError(f"Sequence length ({seq_length}) is greater than or equal to the number of samples ({X.shape[0]}).")
    
        self.seq_length = seq_length
        self.X = []
        self.y = []
    
        # Build sequences
        for i in range(len(X) - seq_length):
            x_seq = X[i : i + seq_length]
            y_label = y[i + seq_length - 1]
            self.X.append(x_seq)
            self.y.append(y_label)
    
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional  # Enable bidirectionality
        )
        
        # Adjust the input size of fc1 based on bidirectionality
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc1 = nn.Linear(lstm_output_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: [batch_size, seq_length, hidden_size * num_directions]
        out = self.fc1(out[:, -1, :])  # Take the output from the last time step
        out = self.relu(out)
        out = self.fc2(out)
        return out


def build_lstm_model(input_size, hidden_size, num_layers, dropout, num_classes, bidirectional=False):
    return LSTMModel(input_size, hidden_size, num_layers, dropout, num_classes, bidirectional)


def train_lstm_model(X_train_2d, y_train_1d, X_val_2d, y_val_1d, feature_names):
    logging.info(f"Starting LSTM training with training data X_train shape: {X_train_2d.shape}, y_train shape: {y_train_1d.shape}")
    logging.info(f"Validation data X_val shape: {X_val_2d.shape}, y_val shape: {y_val_1d.shape}")

    if X_train_2d.shape[0] <= 0 or X_val_2d.shape[0] <= 0:
        logging.error("No data available for LSTM training or validation.")
        return None

    try:
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_2d)
        X_val_scaled = scaler.transform(X_val_2d)

        seq_length = getattr(config, 'lstm_sequence_length', 20)  # Adjust as needed
        train_dataset = LSTMWindowedDataset(X_train_scaled, y_train_1d, seq_length=seq_length)
        val_dataset = LSTMWindowedDataset(X_val_scaled, y_val_1d, seq_length=seq_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.lstm_batch_size,
            shuffle=True,
            num_workers=4,  # Adjust based on your CPU
            pin_memory=False  # If using GPU
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.lstm_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"LSTM training on {device}. Using seq_length={seq_length}.")

        hidden_size = getattr(config, 'lstm_hidden_size', 128)
        num_layers = getattr(config, 'lstm_num_layers', 3)
        dropout = getattr(config, 'lstm_dropout', 0.3)
        num_classes = len(np.unique(y_train_1d))
        bidirectional = getattr(config, 'lstm_bidirectional', True)  # New config parameter

        model = build_lstm_model(
            input_size=X_train_scaled.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            bidirectional=bidirectional  # Pass bidirectionality
        ).to(device)

        # Handle class imbalance
        class_counts = Counter(y_train_1d)
        class_weights = torch.tensor([sum(class_counts.values()) / class_counts[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        lr = getattr(config, 'lstm_learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay

        # Initialize scheduler without the verbose parameter
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=1e-5,
            eps=1e-8
        )

        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = getattr(config, 'lstm_patience', 10)  # Updated patience
        epochs = getattr(config, 'lstm_epochs', 50)  # Increased epochs

        # Ensure the model directory exists
        model_dir = config.BASE_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "lstm_model.pth"

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_accum = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss_accum += loss.item()

            avg_train_loss = train_loss_accum / len(train_loader)

            model.eval()
            val_loss_accum = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss_accum += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_val_loss = val_loss_accum / len(val_loader)
            val_accuracy = correct / total

            logging.info(f"[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.4f}, "
                         f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Scheduler step with avg_val_loss
            scheduler.step(avg_val_loss)

            # Log the current learning rate
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            logging.info(f"Current Learning Rate: {current_lr}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                logging.info(f"Best LSTM model saved at {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logging.info("LSTM early stopping triggered.")
                    break

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
            model.eval()
            logging.info("Best LSTM model loaded from disk.")
            return model, scaler  # Return scaler for consistent preprocessing during inference
        else:
            logging.error("No LSTM model checkpoint found. Returning None.")
            return None

    except Exception as e:
        logging.error(f"Error training LSTM model: {e}", exc_info=True)
        return None



def predict_lstm(lstm_model, scaler, X_2d):
    """
    Inference for LSTM with windowed approach.
    This function aligns predictions to the last index of each sequence and handles padding intelligently.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)
    lstm_model.eval()

    seq_length = getattr(config, 'lstm_sequence_length', 20)  # Ensure consistency with training

    # Scale the input data using the scaler fitted during training
    X_scaled = scaler.transform(X_2d)

    # Build sequences
    if len(X_scaled) < seq_length:
        logging.warning("Not enough data to form a single LSTM window. Returning all HOLDs.")
        dummy_preds = np.ones(shape=(len(X_scaled),), dtype=np.int64)  # e.g., '1' => HOLD
        dummy_probs = np.zeros((len(X_scaled), lstm_model.fc2.out_features), dtype=np.float32)
        return dummy_preds, dummy_probs

    sequences = []
    for i in range(len(X_scaled) - seq_length + 1):
        seq = X_scaled[i:i + seq_length]
        sequences.append(seq)
    sequences = np.array(sequences, dtype=np.float32)  # shape (#windows, seq_length, features)

    # Model inference
    with torch.no_grad():
        batch_input = torch.tensor(sequences, dtype=torch.float32).to(device)
        outputs = lstm_model(batch_input)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # Initialize final predictions and probabilities
    num_classes = lstm_model.fc2.out_features
    final_preds = np.ones((len(X_scaled),), dtype=np.int64)  # default = 1 => HOLD
    final_probs = np.zeros((len(X_scaled), num_classes), dtype=np.float32)

    # Assign predictions to the last index of each window
    for window_idx in range(len(preds)):
        last_index = window_idx + seq_length - 1
        final_preds[last_index] = preds[window_idx]
        final_probs[last_index] = probs[window_idx]

    # Optionally, handle the initial `seq_length - 1` indices
    # For example, carry forward the first prediction
    for i in range(seq_length - 1):
        final_preds[i] = final_preds[seq_length - 1]
        final_probs[i] = final_probs[seq_length - 1]

    return final_preds, final_probs

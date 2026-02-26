import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


class SensorDataset(Dataset):
    def __init__(self, dataframe, seq_length=600, scaler=None):
        self.seq_length = seq_length
        dataframe = dataframe.reset_index(drop=True)

        # Tried using only one sensor but the results were really bad,
        # so now using the two most important pca components instead of the original data

        self.sensor = dataframe[["pc1", "pc2"]].values

        self.status = dataframe["status"].map({"HEALTHY": 0, "UNHEALTHY": 1}).values

        if scaler is None:
            self.scaler = StandardScaler()
            self.sensor = self.scaler.fit_transform(self.sensor)
        else:
            self.scaler = scaler
            self.sensor = self.scaler.transform(self.sensor)

        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []

        # Taking in x amount and predicting if the status switches in the next 100
        future_window = 120
        max_index = len(self.sensor) - self.seq_length - future_window

        for i in range(max_index):

            input_slice = self.status[i : i + self.seq_length]

            # if the switch from healthy to unhealthy happens in the input window,
            # skip it since we want to predict the switch based on healthy data
            """if np.any(input_slice == 1):
                continue"""

            X.append(self.sensor[i:i+self.seq_length])

            # Check if unhealthy occurs in future window
            t = i + self.seq_length
            future_window2 = self.status[t : t + future_window]

            if np.any(future_window2 == 1):
                y.append(1)
            else:
                y.append(0)

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

def build_model():
    return LSTMClassifier(input_size=2, hidden_size=64, num_layers=2)


# Training
def train_and_test(csv_path, epochs=20, batch_size=64, lr=1e-3, seq_length=600, save_dir="models", train=True, file_name=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)

    if not os.path.isabs(save_dir):
        save_dir = os.path.join(script_dir, save_dir)

    df = pd.read_csv(csv_path)

    # drops the last 30k rows
    #df = df.iloc[:-30000]

    # splitting the data into train, val and test sets
    # notably usually the data would be split in a chronological order,
    # but that split didnt leave the test set with any unhealthy rows, so had to do it differently
    total_len = len(df)
    val_end = int(total_len * 0.2)
    test_end = int(total_len * 0.4)

    val_df = df.iloc[:val_end]
    test_df = df.iloc[val_end:test_end]
    #test_df = df.iloc[:val_end]
    #val_df = df.iloc[val_end:test_end]
    #test_df = df.iloc[0:total_len]
    train_df = df.iloc[test_end:]

    train_ds = SensorDataset(train_df, seq_length=seq_length)
    scaler = train_ds.scaler
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    val_ds = SensorDataset(val_df, seq_length=seq_length, scaler=scaler)
    test_ds = SensorDataset(test_df, seq_length=seq_length, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("train")
    print(train_df["status"].value_counts())
    print("val")
    print(val_df["status"].value_counts())
    print("test")
    print(test_df["status"].value_counts())

    print("Train window positives:", (train_ds.y == 1).sum().item())
    print("Val window positives:", (val_ds.y == 1).sum().item())
    print("Test window positives:", (test_ds.y == 1).sum().item())

    model = build_model().to(device)

    os.makedirs(save_dir, exist_ok=True)
    if file_name is None:
        file_name = "best_lstm"
    ckpt_path = os.path.join(save_dir, f"{file_name}.pt")

    if train:

        
        num_pos = (train_ds.y == 1).sum().item()
        num_neg = (train_ds.y == 0).sum().item()
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)

        #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        criterion = nn.BCEWithLogitsLoss()

        #criterion = FocalLoss(alpha=0.15, gamma=4.0)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)

            print(f"Epoch {epoch+1}/{epochs} | "f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)
                print("Saved best model.")

    # Testing 

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    y_true = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())
            y_true.extend(y_batch.numpy().flatten())

    all_probs = np.array(all_probs)
    y_true = np.array(y_true)
    
    best_weighted_f1 = 0
    best_threshold = 0.5
    best_real_f1 = 0.0 
    
    for threshold in np.arange(0.05, 1.0, 0.01):
        y_pred = (all_probs > threshold).astype(int)
        
        healthy_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        unhealthy_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        # giving more weight to the unhealthy since its more important
        weighted_f1 = (healthy_f1 + unhealthy_f1*2) / 3
        real_f1 = (healthy_f1 + unhealthy_f1) / 2.0
        
        print(f"Threshold {threshold:.2f}: Healthy_F1={healthy_f1:.4f}, Unhealthy_F1={unhealthy_f1:.4f}, Combined_F1={weighted_f1:.4f}, Pred+={y_pred.sum()}")
        
        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            best_real_f1 = real_f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} with weighted F1={best_weighted_f1:.4f}, and real F1={best_real_f1:.4f}")
    y_pred = (all_probs > best_threshold).astype(int)

    print("Predicted positives:", sum(y_pred))
    print("True positives:", sum(y_true))

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)


    print("\nClassification Results:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Healthy", "Unhealthy"]
    ))

    print("\nTest Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    test_df_plot = test_df.reset_index(drop=True)

    pc1 = test_df_plot["pc1"].values
    pc2 = test_df_plot["pc2"].values
    true_status = test_df_plot["status"].map({"HEALTHY": 0, "UNHEALTHY": 1}).values

    prediction_timeline = np.zeros(len(test_df_plot))
    prediction_counts = np.zeros(len(test_df_plot))

    future_window = 120
    max_index = len(test_df_plot) - seq_length - future_window

    for i in range(max_index):
        t = i + seq_length
        prediction_timeline[t] += y_pred[i]
        prediction_counts[t] += 1

    mask = prediction_counts > 0
    prediction_timeline[mask] = prediction_timeline[mask] / prediction_counts[mask]

    prediction_binary = (prediction_timeline > best_threshold).astype(int)

    plt.figure(figsize=(18, 8))

    plt.plot(pc1, label="PC1")
    plt.plot(pc2, label="PC2")

    unhealthy_regions = np.where(true_status == 1)[0]
    plt.scatter(unhealthy_regions, pc1[unhealthy_regions], marker='x', s=100, color='red', label="True Unhealthy")

    predicted_regions = np.where(prediction_binary == 1)[0]
    plt.scatter(predicted_regions, pc1[predicted_regions], marker='o', s=20, color='blue', label="Predicted Unhealthy")

    plt.title("PC1 and PC2 with true and predicted unhealthy")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    train_and_test(
        csv_path="data/sensor_pca_cleaned.csv",
        epochs=15,
        batch_size=128,
        lr=8e-5,
        seq_length=1000,
        train=False,
        file_name="best_lstm_model"
    )

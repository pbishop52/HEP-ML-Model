import pandas as pd
import os
import zipfile
import numpy as np
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.optim as optim



downloads_path = os.path.expanduser("~/Downloads")
train_csv_path = os.path.join(downloads_path, "hepmass_extracted", "1000_train.csv.gz")
test_csv_path = os.path.join(downloads_path, "hepmass_extracted", "1000_test.csv.gz")
pd.set_option('display.max_columns', None)

#train_df = pd.read_csv(train_csv_path, compression="gzip", skiprows = 1)
#test_df = pd.read_csv(test_csv_path, compression="gzip", skiprows = 1)
#columns = ['label'] + [f'f{i}' for i in range(27)]
#train_df.columns = columns
#test_df.columns = columns
#train_df.to_parquet(os.path.join(downloads_path, "hepmass_extracted", "1000_train.parquet"))
#test_df.to_parquet(os.path.join(downloads_path, "hepmass_extracted", "1000_test.parquet"))
#print("hello")
train_df = pd.read_parquet(os.path.join(downloads_path, "hepmass_extracted", "1000_train.parquet"))
test_df = pd.read_parquet(os.path.join(downloads_path, "hepmass_extracted", "1000_test.parquet"))

x_train = train_df.drop('label', axis=1).values
y_train  =train_df['label'].values
x_test = test_df.drop('label', axis=1).values
y_test  =test_df['label'].values

x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024)




class HEPMASSClassifier(nn.Module):
    def __init__(self):
        super(HEPMASSClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(27, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary classification output
        )

    def forward(self, x):
        return self.model(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HEPMASSClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
if __name__ == "__main__":

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)
            total_train += y_batch.size(0)

        avg_train_loss = total_train_loss / total_train

        # --- Evaluation on Test Set (also no threshold inside loop) ---
        model.eval()
        total_test_loss = 0.0
        all_test_outputs = []
        all_test_labels = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                total_test_loss += loss.item() * x_batch.size(0)
                all_test_outputs.append(outputs.cpu())
                all_test_labels.append(y_batch.cpu())

        avg_test_loss = total_test_loss / len(test_loader.dataset)

        # Evaluate accuracy (optional): use threshold *after* loop
        outputs_cat = torch.cat(all_test_outputs)
        labels_cat = torch.cat(all_test_labels)
        predictions = (outputs_cat >= 0.5).float()  # Apply threshold only here
        test_accuracy = (predictions == labels_cat).float().mean().item()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_no_thresh_{epoch+1}.pt"))


"""
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # shape: (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)

            predictions = (outputs >= 0.5).float()
            correct_train += (predictions == y_batch).sum().item()
            total_train += y_batch.size(0)

        avg_train_loss = total_train_loss / total_train
        train_accuracy = correct_train / total_train

        # --- Evaluation on Test Set ---
        model.eval()
        total_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                total_test_loss += loss.item() * x_batch.size(0)

                predictions = (outputs >= 0.5).float()
                correct_test += (predictions == y_batch).sum().item()
                total_test += y_batch.size(0)

        avg_test_loss = total_test_loss / total_test
        test_accuracy = correct_test / total_test

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_no_thresh_{epoch+1}.pt"))

"""
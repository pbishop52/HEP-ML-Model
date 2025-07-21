import pandas as pd
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import torch.optim as optim
from test import HEPMASSClassifier
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

start_time = time.time()

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

model = HEPMASSClassifier()
model.load_state_dict(torch.load("checkpoints/epoch_no_thresh_5.pt", map_location=torch.device("cpu")))
model.eval()

all_preds = []
all_labels = []

start_time = time.time()

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        preds = outputs.squeeze().numpy()   # Get probabilities
        labels = y_batch.numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

print(f"\nInference completed in {time.time() - start_time:.2f} seconds")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

"""
plt.figure(dpi=300)
plt.hist(all_preds[all_labels == 1], bins=100, alpha=0.6, label="Signal (1)", color = 'skyblue')
plt.hist(all_preds[all_labels == 0], bins=100, alpha=0.6, label="Background (0)", color = 'red')
plt.xlabel("Model Output Probability")
plt.ylabel("Frequency")
plt.legend()
plt.title("Model Output for Signal vs Background")
plt.tick_params(direction = "in")
plt.show()
"""
"""
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
thresholds = [0.1,0.3,0.5,0.7,0.9]

for t in thresholds:
    thresholded_preds = (all_preds >= t).astype(int)

    plt.hist(thresholded_preds[all_labels == 1], bins=100, alpha=0.6, label= f"Signal (threshold={t})")
    plt.hist(thresholded_preds[all_labels == 0], bins=100, alpha=0.6, label=f"Background (threshold={t})")
plt.xticks([0,1],["Predicted Background","Predicted Signal"])
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.legend()
plt.title("Model Output at Different Thresholds")
plt.grid(True)
plt.tight_layout()
plt.show()
"""


"""
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure(dpi=300)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}",color ='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend()
plt.tick_params(direction="in")
plt.show()
"""
#"""
threshold = 0.9
preds = (all_preds >= threshold).astype(int)

cm = confusion_matrix(all_labels, preds)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="YlOrRd")

labels = np.array([["TN", "FP"], ["FN", "TP"]])
counts = np.array([[tn, fp], [fn, tp]])

for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i, j]}\n{counts[i, j]}", ha="center", va="center", fontsize=12, fontweight='bold')

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Background", "Signal"])
ax.set_yticklabels(["Background", "Signal"])
plt.title(f"Confusion Matrix at Threshold {threshold}")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
#"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


threshold = 0.9
binary_preds = (all_preds >= threshold).astype(int)

accuracy = accuracy_score(all_labels, binary_preds)
precision = precision_score(all_labels, binary_preds)
recall = recall_score(all_labels, binary_preds)
f1 = f1_score(all_labels, binary_preds)

print(f"Threshold: {threshold}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Using SMOTE instead of random oversampling

# 1. Load the dataset
data = pd.read_csv('cleaned_breast_cancer_data.csv')

# ------------------------------
# Visualize Survival Outcome (for context)
# ------------------------------
status_labels = {1: "Alive", 2: "Dead"}
data['Status_Label'] = data['Status'].map(status_labels)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
data['Status_Label'].value_counts().plot(kind='bar', edgecolor='black')
plt.title("Survival Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(data['Survival Months'], bins=20, edgecolor='black')
plt.title("Survival Months Distribution")
plt.xlabel("Survival Months")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ------------------------------
# Check and Convert Class Labels
# ------------------------------
print("Original class distribution (Status before conversion):")
print(data['Status'].value_counts())  # e.g., 3408 Alive, 616 Dead

# Convert Status from {1,2} to {0,1} (0: Alive, 1: Dead)
data['Status'] = data['Status'] - 1

# ------------------------------
# Prepare Data for Modeling
# ------------------------------
features_to_exclude = ['Status', 'Survival Months', 'Status_Label']
X = data.drop(columns=features_to_exclude).values
y = data['Status'].values

# Split data into training (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ------------------------------
# Oversample using SMOTE
# ------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE, training class distribution:", np.bincount(y_train_res))

# Convert oversampled training data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_res)
y_train_tensor = torch.LongTensor(y_train_res)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ------------------------------
# Define Focal Loss
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        alpha: weighting factor for the rare class.
        gamma: focusing parameter for hard vs easy examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross entropy loss without reduction
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# ------------------------------
# Define the Neural Network Model with Dropout
# ------------------------------
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3, output_dim=2):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

input_dim = X_train.shape[1]
model = FeedForwardNN(input_dim=input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3, output_dim=2)

# ------------------------------
# Set up Loss Function and Optimizer
# ------------------------------
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ------------------------------
# Training Loop with Early Stopping
# ------------------------------
num_epochs = 50
best_val_loss = np.inf
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            outputs = model(val_X)
            loss = criterion(outputs, val_y)
            val_loss += loss.item() * val_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    scheduler.step(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        best_model_state = model.state_dict()
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break

model.load_state_dict(best_model_state)

# ------------------------------
# Evaluate on Test Set
# ------------------------------
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for test_X, test_y in test_loader:
        outputs = model(test_X)
        loss = criterion(outputs, test_y)
        test_loss += loss.item() * test_X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_test += test_y.size(0)
        correct_test += (predicted == test_y).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_y.cpu().numpy())

avg_test_loss = test_loss / len(test_loader.dataset)
test_acc = correct_test / total_test
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Alive", "Dead"]))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

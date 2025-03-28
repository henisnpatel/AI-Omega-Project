import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the dataset
data = pd.read_csv('cleaned_breast_cancer_data.csv')

# Assuming 'Status' is the target column where 1=Alive and 2=Dead,
# we convert it to 0 and 1 for binary classification.
data['Status'] = data['Status'] - 1  # Now, 0 indicates Alive, 1 indicates Dead

# Separate features and target
X = data.drop('Status', axis=1).values
y = data['Status'].values

# 2. Split the data into training, validation, and test sets
# For instance, 70% training, 15% validation, 15% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 3. Standardize the feature values (important for neural network convergence)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader objects for batching
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# 4. Define a feed-forward neural network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=2):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Initialize the model. The input dimension is the number of features.
input_dim = X_train.shape[1]
model = FeedForwardNN(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=2)

# 5. Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()       # Clear gradients from the previous iteration
        outputs = model(batch_X)      # Forward pass
        loss = criterion(outputs, batch_y)
        loss.backward()             # Backpropagation
        optimizer.step()            # Update weights
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Evaluate on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            outputs = model(val_X)
            _, predicted = torch.max(outputs.data, 1)
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

print("Training complete.")

# 7. Optional: Evaluate on the test set
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for test_X, test_y in test_loader:
        outputs = model(test_X)
        _, predicted = torch.max(outputs.data, 1)
        total_test += test_y.size(0)
        correct_test += (predicted == test_y).sum().item()
test_acc = correct_test / total_test
print(f"Test Accuracy: {test_acc:.4f}")
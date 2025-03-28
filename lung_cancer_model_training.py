import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the dataset
data = pd.read_csv('cleaned_lung_cancer_data.csv')

# Drop unnecessary columns: 'index' and 'Patient Id' are identifiers, not features
X = data.drop(['index', 'Patient Id', 'Level'], axis=1).values  # Features

y = data['Level'].values  # Target variable (Level of disease)

# 2. Split the data into training, validation, and test sets
# 70% Training, 15% Validation, 15% Test
test_size = 0.3  # 30% of data will be used for testing+validation
val_test_ratio = 0.5  # 50% of test_size is validation, 50% is testing

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_test_ratio, random_state=42,
                                                stratify=y_temp)

# 3. Standardize the feature values (Important for stable neural network training)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data into PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 4. Create DataLoader objects for batch processing (Efficient training)
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 5. Define the Neural Network Model
class LungCancerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=3):  # 3 output classes
        super(LungCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Initialize the model
input_dim = X_train.shape[1]  # Number of features
model = LungCancerNN(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=3)

# 6. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer for better convergence

# 7. Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # 8. Evaluate on Validation Set
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            outputs = model(val_X)
            _, predicted = torch.max(outputs.data, 1)  # Get class with the highest probability
            total += val_y.size(0)
            correct += (predicted == val_y).sum().item()
    val_acc = correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

print("Training complete.")

# 9. Final Evaluation on Test Set
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

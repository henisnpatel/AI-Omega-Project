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

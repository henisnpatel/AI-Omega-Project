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

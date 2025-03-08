# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = "data_1.csv"  # Update with correct path if needed
df = pd.read_csv(file_path)

# Fix column names (remove trailing spaces)
df.columns = df.columns.str.strip()

# Display dataset info
print(f"Dataset Shape: {df.shape}")
print("First 5 Rows:\n", df.head())
print("Available Columns in Dataset:\n", df.columns)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Drop unnecessary columns
drop_cols = ['pkSeqID', 'stime', 'saddr', 'daddr', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Fill missing values in numeric columns with median
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing values in categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])


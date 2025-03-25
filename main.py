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

# Convert non-numeric columns in 'sport' and 'dport' to numeric (errors='coerce' turns invalid values to NaN)
df['sport'] = pd.to_numeric(df['sport'], errors='coerce')
df['dport'] = pd.to_numeric(df['dport'], errors='coerce')

# Fill missing values in 'sport' and 'dport' columns with median
df['sport'] = df['sport'].fillna(df['sport'].median())
df['dport'] = df['dport'].fillna(df['dport'].median())

# Fill missing values in categorical columns with mode or a default value if mode is empty
def fill_missing_mode(column):
    mode_value = column.mode()
    if mode_value.empty:  # If mode is empty, use 'Unknown' as a default value
        return column.fillna('Unknown')
    else:
        return column.fillna(mode_value[0])

# Apply the fill_missing_mode function to the categorical columns
df['smac'] = fill_missing_mode(df['smac'])
df['dmac'] = fill_missing_mode(df['dmac'])
df['soui'] = fill_missing_mode(df['soui'])
df['doui'] = fill_missing_mode(df['doui'])
df['sco'] = fill_missing_mode(df['sco'])
df['dco'] = fill_missing_mode(df['dco'])

# Drop unnecessary columns (saddr and daddr are dropped here)
drop_cols = ['pkSeqID', 'stime', 'ltime', 'seq', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco', 'saddr', 'daddr']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode categorical columns ('flgs', 'proto', 'state', 'category', 'subcategory')
categorical_cols = ['flgs', 'proto', 'state', 'category', 'subcategory']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Separate features (X) and target labels (y)
X = df.drop('category', axis=1)  # 'category' is the target column
y = df['category']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features to have zero mean and unit variance
# Exclude non-numeric columns (like 'saddr', 'daddr')
numeric_cols = X.select_dtypes(include=[np.number]).columns
X_train_scaled = X_train[numeric_cols]
X_test_scaled = X_test[numeric_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

# Now include the non-numeric columns back
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

//


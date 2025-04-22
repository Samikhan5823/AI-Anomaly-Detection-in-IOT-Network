# AI Anomaly Detection 
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn ML modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# SMOTE for class balancing
from imblearn.over_sampling import SMOTE

# Load Dataset
file_path = "data_1.csv"
df = pd.read_csv(file_path)

# Fix Column Names
df.columns = df.columns.str.strip()

# Preview Dataset
print(f"Dataset Shape: {df.shape}")
print("First 5 Rows:\n", df.head())
print("Columns:\n", df.columns)

# Check Missing Values
print("Missing Values:\n", df.isnull().sum())

# Convert Port Columns to Numeric
df['sport'] = pd.to_numeric(df['sport'], errors='coerce')
df['dport'] = pd.to_numeric(df['dport'], errors='coerce')
df['sport'] = df['sport'].fillna(df['sport'].median())
df['dport'] = df['dport'].fillna(df['dport'].median())

# Fill Missing Categorical Columns
def fill_missing_mode(column):
    mode_value = column.mode()
    return column.fillna(mode_value[0] if not mode_value.empty else 'Unknown')

for col in ['smac', 'dmac', 'soui', 'doui', 'sco', 'dco']:
    df[col] = fill_missing_mode(df[col])

# Drop Irrelevant Columns
drop_cols = ['pkSeqID', 'stime', 'ltime', 'seq', 'smac', 'dmac', 'soui', 'doui', 
             'sco', 'dco', 'saddr', 'daddr']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode Categorical Columns
categorical_cols = ['flgs', 'proto', 'state', 'category', 'subcategory']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Define Features and Target
X = df.drop('category', axis=1)
y = df['category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize Numeric Features
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled = scaler.transform(X_test[numeric_cols])
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols)

# Visualize Class Imbalance
plt.figure(figsize=(8,4))
sns.countplot(x=y)
plt.title("Before SMOTE - Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Visualize After SMOTE
plt.figure(figsize=(8,4))
sns.countplot(x=y_train_res)
plt.title("After SMOTE - Balanced Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Heatmap
corr = X_train_res.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Optimized PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_res)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train_res, palette='viridis', alpha=0.7)
plt.title("PCA of Network Traffic Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# Decode Label Classes
print("Decoded Label Classes:\n", encoder.inverse_transform(np.unique(y)))

# Initialize Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train Models
rf_model.fit(X_train_res, y_train_res)
svm_model.fit(X_train_res, y_train_res)
mlp_model.fit(X_train_res, y_train_res)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Evaluate Models
def evaluate_model(name, y_true, y_pred):
    print(f"\n====== {name} Evaluation ======")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("MLP Classifier", y_test, y_pred_mlp)

# Feature Importance - Random Forest
importances = rf_model.feature_importances_
feature_names = X_train_scaled.columns
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_df.head(15))
plt.title("Top 15 Important Features - Random Forest")
plt.tight_layout()
plt.show()

# Extra: Permutation Importance 
perm_importance = permutation_importance(rf_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    "Feature": X_test_scaled.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=perm_df.head(15))
plt.title("Permutation Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# Optional: Hyperparameter Tuning for SVM (quick example)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=3)
grid_svm.fit(X_train_res, y_train_res)
print("\nBest SVM Parameters:", grid_svm.best_params_)

# Optional: Save Model (if needed for deployment)
import joblib
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(mlp_model, 'mlp_model.pkl')

# end

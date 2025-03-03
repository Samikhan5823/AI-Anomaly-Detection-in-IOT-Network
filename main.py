# Import necessary libraries
import pandas as pd
import numpy as np

# Loading the Bot-IoT dataset
file_path = "data_1.csv" 
df = pd.read_csv(file_path)

# Displaying dataset structure
print("Dataset Shape:", df.shape)
print(df.head())

# Dropping irrelevant columns 
columns_to_drop = ['pkSeqID', 'saddr', 'daddr', 'stime', 'category', 'subcategory']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical labels for attack detection
label_encoder = LabelEncoder()
df['attack'] = label_encoder.fit_transform(df['attack']) 

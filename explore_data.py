import pandas as pd

# Paths to the provided dataset
train_path = "dataset/train/train.csv"
test_path = "dataset/test/test.csv"

# Load the data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Basic information
print("\n===== TRAIN DATA =====")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())

print("\nFirst 5 training rows:")
print(train_df.head())

print("\n===== TEST DATA =====")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())

print("\nFirst 5 test rows:")
print(test_df.head())

# Check missing values
print("\n===== MISSING VALUES =====")
print("Train:")
print(train_df.isnull().sum())

print("\nTest:")
print(test_df.isnull().sum())

# Check sentiment distribution
if "label" in train_df.columns:
    print("\n===== LABEL DISTRIBUTION =====")
    print(train_df["label"].value_counts())
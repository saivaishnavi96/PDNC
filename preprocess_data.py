import pandas as pd

# Load training data
train_path = "dataset/train/train.csv"
train_df = pd.read_csv(train_path)

print("Original training shape:", train_df.shape)

# Check missing aspect terms
print("\nMissing aspect terms:")
print(train_df["aspect_term"].isnull().sum())

# Remove rows where the aspect term is missing
train_df = train_df.dropna(subset=["aspect_term"]).copy()

print("\nShape after removing missing aspect terms:", train_df.shape)

# Check labels
print("\nLabel distribution:")
print(train_df["label"].value_counts())

# Convert labels to numbers
label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

train_df["label_id"] = train_df["label"].map(label_map)

print("\nLabel mapping:")
print(label_map)

print("\nFirst 5 processed rows:")
print(
    train_df[
        ["text", "aspect_term", "context_with_aspect_tag", "label", "label_id"]
    ].head()
)
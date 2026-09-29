import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PATH = "./dataset/train/train.csv"
TEST_PATH = "./dataset/test/test.csv"

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "./trained_model"
CHECKPOINT_DIR = "./checkpoints"

MAX_LENGTH = 128
NUM_LABELS = 3

LABEL_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Original train shape:", train_df.shape)

# Remove the one row with missing aspect
train_df = train_df.dropna(subset=["aspect_term"]).copy()

# Convert labels to numbers
train_df["label_id"] = train_df["label"].map(LABEL_MAP)

# Safety check
train_df = train_df.dropna(subset=["label_id"]).copy()
train_df["label_id"] = train_df["label_id"].astype(int)

print("After cleaning:", train_df.shape)


# ============================================================
# ASPECT + REVIEW INPUT
# ============================================================

# The model receives:
#
# [CLS] aspect [SEP] review [SEP]
#
# The tokenizer creates the actual special tokens.

train_df["aspect_input"] = (
    train_df["aspect_term"]
    .astype(str)
    .str.strip()
)

train_df["review_input"] = (
    train_df["text"]
    .astype(str)
    .str.strip()
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_part, val_part = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    stratify=train_df["label_id"],
)

train_part = train_part.reset_index(drop=True)
val_part = val_part.reset_index(drop=True)

print()
print("Training samples:", len(train_part))
print("Validation samples:", len(val_part))

print()
print("Training label distribution:")
print(
    train_part["label_id"]
    .value_counts()
    .sort_index()
)

print()
print("Validation label distribution:")
print(
    val_part["label_id"]
    .value_counts()
    .sort_index()
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_dataframe(df):

    return tokenizer(
        df["aspect_input"].tolist(),
        df["review_input"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


print("Tokenizing training data...")

train_tokens = tokenize_dataframe(
    train_part
)

print("Tokenizing validation data...")

val_tokens = tokenize_dataframe(
    val_part
)

print("Tokenization complete.")


# ============================================================
# PYTORCH DATASET
# ============================================================

class ABSADataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(
                value[idx],
                dtype=torch.long
            )
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


train_dataset = ABSADataset(
    train_tokens,
    train_part["label_id"].values
)

val_dataset = ABSADataset(
    val_tokens,
    val_part["label_id"].values
)

print()
print("PyTorch datasets created.")
print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

print()
print("Loading DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_MAP,
)

print("Model loaded.")
print("Number of classes:", NUM_LABELS)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print()
print("=" * 60)
print("CREATING TRAINING CONFIGURATION")
print("=" * 60)

training_args = TrainingArguments(

    # Where temporary checkpoints are stored
    output_dir=CHECKPOINT_DIR,

    # Train on all 6399 training examples
    num_train_epochs=2,

    # CPU-friendly batch sizes
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    # Learning rate
    learning_rate=2e-5,

    # Regularization
    weight_decay=0.01,

    # Warmup
    # Using warmup_steps instead of warmup_ratio
    # because of the installed Transformers version.
    warmup_steps=100,

    # Evaluate after every epoch
    eval_strategy="epoch",

    # Save after every epoch
    save_strategy="epoch",

    # Keep the best Macro-F1 model
    load_best_model_at_end=True,

    metric_for_best_model="eval_macro_f1",

    greater_is_better=True,

    # Logging
    logging_steps=50,

    # Keep only the last two checkpoints
    save_total_limit=2,

    # Disable external logging
    report_to="none",

    # CPU machine
    dataloader_pin_memory=False,

    # Avoid unnecessary multiprocessing issues on Windows
    dataloader_num_workers=0,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1
        )
    ],
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING IMPROVED TRAINING")
print("=" * 60)

print()
print("Training configuration:")
print("- Model: DistilBERT")
print("- Training samples: 6399")
print("- Validation samples: 1600")
print("- Epochs: 2")
print("- Maximum sequence length:", MAX_LENGTH)
print("- Batch size: 16")
print("- Learning rate: 2e-5")
print("- Primary metric: Macro F1")
print()

train_result = trainer.train()


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

results = trainer.evaluate()

validation_accuracy = results.get(
    "eval_accuracy",
    0.0
)

validation_macro_f1 = results.get(
    "eval_macro_f1",
    0.0
)

print()
print(
    f"Validation Accuracy: "
    f"{validation_accuracy:.4f}"
)

print(
    f"Validation Macro F1: "
    f"{validation_macro_f1:.4f}"
)

print()
print(
    "These are LOCAL validation results, "
    "not leaderboard results."
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

print()
print("=" * 60)
print("SAVING BEST MODEL")
print("=" * 60)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

trainer.save_model(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)

print()
print("Best model saved to:")
print(MODEL_DIR)


# ============================================================
# PREPARE TEST DATA
# ============================================================

print()
print("=" * 60)
print("CREATING TEST PREDICTIONS")
print("=" * 60)

test_aspects = (
    test_df["aspect_term"]
    .fillna("")
    .astype(str)
    .str.strip()
)

test_reviews = (
    test_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# TOKENIZE TEST DATA
# ============================================================

print("Tokenizing test data...")

test_tokens = tokenizer(
    test_aspects.tolist(),
    test_reviews.tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
)

print("Test data tokenized.")


# ============================================================
# TEST DATASET
# ============================================================

class TestDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):

        self.encodings = encodings

    def __len__(self):

        return len(
            self.encodings["input_ids"]
        )

    def __getitem__(self, idx):

        return {
            key: torch.tensor(
                value[idx],
                dtype=torch.long
            )
            for key, value in self.encodings.items()
        }


test_dataset = TestDataset(
    test_tokens
)

print(
    "Test dataset size:",
    len(test_dataset)
)


# ============================================================
# PREDICT TEST DATA
# ============================================================

print()
print("Generating test predictions...")

prediction_output = trainer.predict(
    test_dataset
)

test_logits = prediction_output.predictions

test_predicted_ids = np.argmax(
    test_logits,
    axis=-1
)

test_predicted_labels = [
    ID_TO_LABEL[int(prediction)]
    for prediction in test_predicted_ids
]


# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "id": test_df["id"],

    "prediction": test_predicted_labels,

})


# ============================================================
# SAVE SUBMISSION
# ============================================================

os.makedirs(
    "./submission",
    exist_ok=True
)

submission_path = (
    "./submission/submission.csv"
)

submission.to_csv(
    submission_path,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("SUBMISSION CREATED")
print("=" * 60)

print()
print(
    "Submission file:",
    submission_path
)

print(
    "Number of rows:",
    len(submission)
)

print()
print("First 10 predictions:")

print(
    submission.head(10).to_string(
        index=False
    )
)

print()
print("Prediction distribution:")

print(
    submission["prediction"]
    .value_counts()
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
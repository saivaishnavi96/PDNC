# Aspect-Based Sentiment Analysis (ABSA) for Consumer Reviews

> [!NOTE]
> ### AI Assistant Policy & Strict Rules
> - **AI Coding Assistance IS Allowed:** You may use AI coding assistants (ChatGPT, Claude, Cursor, Copilot) as pair programmers to help you write code, learn PyTorch/Transformers syntax, and debug errors.
> - **No Single-Prompt Code Dumps:** Work with your AI co-pilot step-by-step through the roadmap sections in `starter_notebook.ipynb` rather than asking for a monolithic solution.
> - **Mandatory Pitch & Defense:** Teams must present a 2-minute architectural pitch to faculty judges and explain their code. You must thoroughly understand every line you submit!
> - **NO EXTERNAL APIS FOR PREDICTIONS:** Using proprietary inference APIs (OpenAI, Anthropic, Gemini, Groq, etc.) to generate test predictions is **strictly prohibited**. All models must run locally or on Kaggle GPU.

## Description

Customer reviews rarely express a single, monolithic sentiment. A typical consumer evaluation evaluates multiple discrete attributes simultaneously—often with conflicting feelings within the exact same sentence. For example:
> *"The decor is not special at all, but their food and amazing prices make up for it."*

In this single sentence, the customer expresses a **Negative** sentiment toward the `decor`, but a **Positive** sentiment toward `food` and `prices`. Standard sentence-level sentiment classifiers collapse these nuances, assigning an arbitrary aggregate score that obscures critical feedback.

### The Objective
Develop a deep learning architecture capable of **Aspect-Term Sentiment Analysis (ATSA)**: predicting the targeted sentiment polarity (`Positive`, `Negative`, `Neutral`) conditioned specifically on an extracted aspect term span within its surrounding review context.

### Evaluation Metrics
Submissions are evaluated on:
- **Primary Metric:** **Macro-Averaged F1 Score** across all 3 sentiment classes (`Negative`, `Neutral`, `Positive`).
- **Secondary Metric:** **Classification Accuracy**.

---

## Dataset & Directory Structure

```text
01_Aspect_Based_Sentiment_Analysis/
├── README.md
├── starter_notebook.ipynb               # Markdown guide detailing what to implement
├── dataset/
│   ├── train/
│   │   └── train.csv                    # 8,000 samples with ground-truth polarity labels
│   └── test/
│       ├── test.csv                     # 2,000 unique test samples WITHOUT labels
│       └── sample_submission.csv        # 2,000 rows illustrating submission schema
├── submission/
│   └── README.md                        # Drop your team submission CSV here
└── utils/
    └── validator.py                     # Automatic order-checking & submission validator
```

### Data Schema & Field Definitions

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique challenge identifier (`ABSA_TR_00001` for train, `ABSA_TEST_0001` for test). |
| `text` | String | Complete consumer review sentence. |
| `aspect_term` | String | The specific target aspect entity to evaluate (e.g., `decor`, `battery life`, `pasta`). |
| `from_idx` | Integer | 0-indexed starting character offset of `aspect_term` in `text`. |
| `to_idx` | Integer | 0-indexed ending character offset of `aspect_term` in `text`. |
| `context_with_aspect_tag`| String | Review sentence where the target aspect span is replaced by `$T$` (standard ATSA conditioning representation). |
| `label` | String | Ground-truth polarity: `Positive`, `Negative`, or `Neutral` (present only in `train.csv`). |

---

## Running on Kaggle & Submission Instructions

### Step 1: Upload Starter Notebook to Kaggle
1. Open [Kaggle](https://www.kaggle.com/) and upload [`starter_notebook.ipynb`](starter_notebook.ipynb).
2. Under **Notebook Options** in the right-hand panel, set **Accelerator** to **GPU T4 x2**.

### Step 2: Upload Dataset
- Click **Add Input** $\rightarrow$ **Upload Dataset** and upload `dataset/train/train.csv` and `dataset/test/test.csv`.

### Step 3: Implement, Train & Export Predictions
- Follow the markdown instructions in each section of `starter_notebook.ipynb` to build your pipeline.
- Export your test predictions as `submission.csv` with exactly two columns: `['id', 'prediction']`.

### Step 4: Validate Order Locally
1. Drop your downloaded file into `submission/submission.csv`.
2. Run the automated validator:
   ```bash
   python utils/validator.py
   ```
   `validator.py` checks for null values, ensures valid labels, automatically corrects any scrambled row ordering to match the official test sequence, and saves `submission/submission_verified.csv`.

---

## Tips

- **You should try** exploring open-source pretrained transformer backbones such as `microsoft/deberta-v3-small`, `roberta-base`, or `bert-base-uncased`.
- **You should try** conditioning your model by formatting the input as a sequence pair: `[CLS] aspect_term [SEP] sentence [SEP]`.
- **You should try** experimenting with the provided masked representation `$T$` in `context_with_aspect_tag` to focus attention on the aspect slot.
- **You should try** using class weights in your loss function or focal loss to handle the large number of Neutral instances.
- **You should try** learning rate warmup with `AdamW` using a small learning rate (around `2e-5`) and gradient clipping.
- **You should try** monitoring validation Macro F1 rather than standard accuracy, as Macro F1 penalizes models that ignore minority classes.

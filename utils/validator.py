"""
Submission Validator & Evaluator for Problem 01: Aspect-Based Sentiment Analysis (ABSA)
- Automatically checks submission format, column names, missing values, and valid labels.
- Validates and enforces exact ID row ordering matching the official test set.
- Automatically saves the verified & ordered file into the `submission/` folder.
- If run by organizers (with evaluation key), computes official metrics and records in results.csv.

Usage:
    # Student verification:
    python utils/validator.py
    python utils/validator.py --file submission/my_submission.csv

    # Organizer evaluation:
    python utils/validator.py --file submission/my_submission.csv --eval
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
EVAL_KEY_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "organizer_evaluation_keys"))
DEFAULT_EVAL_KEY = os.path.join(EVAL_KEY_DIR, "01_absa_solution.csv")

VALID_LABELS = ['Negative', 'Neutral', 'Positive']
LABEL_MAP = {
    'NEGATIVE': 'Negative', '0': 'Negative', '-1': 'Negative',
    'NEUTRAL': 'Neutral',   '1': 'Neutral',
    'POSITIVE': 'Positive', '2': 'Positive'
}

def normalize_label(val):
    s = str(val).strip().capitalize()
    if s in VALID_LABELS:
        return s
    u = str(val).strip().upper()
    return LABEL_MAP.get(u, s)

def find_candidate_file(explicit_file=None):
    if explicit_file:
        if os.path.exists(explicit_file):
            return explicit_file
        candidate = os.path.join(SUBMISSION_DIR, explicit_file)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Specified submission file not found: {explicit_file}")
    
    if os.path.exists(SUBMISSION_DIR):
        csv_files = [os.path.join(SUBMISSION_DIR, f) for f in os.listdir(SUBMISSION_DIR) if f.endswith('.csv') and f != 'submission_verified.csv']
        if csv_files:
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return csv_files[0]

    sample = os.path.join(DATASET_DIR, "test", "sample_submission.csv")
    if os.path.exists(sample):
        print("[INFO] No file found in submission/ folder. Running verification using dataset/test/sample_submission.csv.")
        return sample
        
    raise FileNotFoundError("No candidate submission CSV found! Please place your file in submission/submission.csv")

def validate_submission(file_path, do_eval=False, solution_key=None):
    print("="*65)
    print("ABSA SUBMISSION VALIDATOR & ORDER VERIFIER")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Submission: {file_path}")
    print("="*65)

    ref_test_path = os.path.join(DATASET_DIR, "test", "test.csv")
    if not os.path.exists(ref_test_path):
        raise FileNotFoundError(f"Reference test file missing: {ref_test_path}")

    ref_test_df = pd.read_csv(ref_test_path)
    expected_ids = ref_test_df['id'].tolist()
    expected_count = len(expected_ids)

    # 1. Read submission
    try:
        sub_df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read submission CSV: {e}")
        return False

    sub_df.columns = [c.strip().lower() for c in sub_df.columns]
    if 'id' not in sub_df.columns or 'prediction' not in sub_df.columns:
        print(f"[ERROR] Missing required columns! Expected ['id', 'prediction'], found {sub_df.columns.tolist()}")
        return False

    # 2. Check nulls
    null_count = sub_df['prediction'].isnull().sum()
    if null_count > 0:
        print(f"[WARNING] Found {null_count} null/missing predictions! Filling with 'Neutral'.")
        sub_df['prediction'] = sub_df['prediction'].fillna('Neutral')

    # 3. Check and correct ordering
    submitted_ids = sub_df['id'].tolist()
    is_ordered = (submitted_ids == expected_ids)
    
    if not is_ordered:
        print("[NOTICE] Submission IDs are out of order or mismatched. Re-aligning to official test sequence...")
        sub_indexed = sub_df.drop_duplicates(subset=['id']).set_index('id')
        missing_ids = set(expected_ids) - set(sub_indexed.index)
        if missing_ids:
            print(f"[WARNING] Submission is missing {len(missing_ids)} expected IDs! Missing IDs will default to 'Neutral'.")
            
        sub_ordered = sub_indexed.reindex(expected_ids, fill_value='Neutral').reset_index()
        sub_ordered.rename(columns={'index': 'id'}, inplace=True)
        sub_df = sub_ordered
        order_corrected = True
        print("[SUCCESS] Successfully re-ordered all rows to match official test sequence.")
    else:
        order_corrected = False
        print("[SUCCESS] Submission row count and ID ordering match test set exactly.")

    # 4. Normalize Predictions
    sub_df['prediction'] = sub_df['prediction'].apply(normalize_label)

    # 5. Save verified submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    verified_path = os.path.join(SUBMISSION_DIR, "submission_verified.csv")
    sub_df[['id', 'prediction']].to_csv(verified_path, index=False)
    print(f"[SAVED] Verified & ordered submission saved to: {verified_path}")
    print(f"[STATUS] Passed validation! Ready for evaluation ({len(sub_df)} rows).")

    # 6. Evaluation (if key is available or requested)
    key_path = solution_key if solution_key else DEFAULT_EVAL_KEY
    if do_eval or (key_path and os.path.exists(key_path)):
        if os.path.exists(key_path):
            print("\n" + "-"*40)
            print("RUNNING BENCHMARK EVALUATION (ORGANIZER MODE):")
            gt_df = pd.read_csv(key_path)
            merged = pd.merge(gt_df, sub_df, on='id', suffixes=('_gt', '_pred'))
            
            y_true = merged['label'].apply(normalize_label)
            y_pred = merged['prediction']

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, labels=VALID_LABELS, zero_division=0)
            
            print(f"  Matched Rows:       {len(merged)} / {expected_count}")
            print(f"  Accuracy:           {acc * 100:.2f}%")
            print(f"  Macro F1 (Primary): {macro_f1 * 100:.2f}%")
            for lbl, f1_val in zip(VALID_LABELS, per_class_f1):
                print(f"  - {lbl:<10} F1:  {f1_val * 100:.2f}%")
            print("-"*40)
            
            # Record in results.csv
            file_exists = os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = pd.DataFrame([{
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'submission_file': os.path.basename(file_path),
                    'matched_samples': len(merged),
                    'total_samples': expected_count,
                    'order_corrected': order_corrected,
                    'accuracy': round(acc, 4),
                    'macro_f1': round(macro_f1, 4),
                    'status': 'PASSED'
                }])
                writer.to_csv(f, header=not file_exists, index=False)
            print(f"[LOGGED] Evaluation score logged to: {RESULTS_CSV}")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate & Order ABSA Submissions")
    parser.add_argument('--file', type=str, default=None, help='Path to submission CSV')
    parser.add_argument('--eval', action='store_true', help='Evaluate if solution key is available')
    parser.add_argument('--solution_key', type=str, default=None, help='Path to solution key CSV')
    args = parser.parse_args()

    try:
        target = find_candidate_file(args.file)
        validate_submission(target, do_eval=args.eval, solution_key=args.solution_key)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

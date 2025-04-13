import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and split the Sleep dataset.")
    parser.add_argument("dataset_input", type=str,
                        help="Path to the cleaned CSV dataset (e.g., data/Sleep_Data_Cleaned.csv)")
    parser.add_argument("all_train_output", type=str,
                        help="Path to output the full cleaned dataset as TSV (e.g., data/full_cleaned.tsv)")
    parser.add_argument("train_output", type=str,
                        help="Path to output the training subset TSV (e.g., data/train.tsv)")
    parser.add_argument("test_output", type=str,
                        help="Path to output the test subset TSV (e.g., data/test.tsv)")
    args = parser.parse_args()

    # Read the new cleaned dataset.
    df = pd.read_csv(args.dataset_input)

    # Binarize target
    target = df.pop("Sleep_Quality_Score")
    threshold = target.median()
    df["Sleep_Quality_Score"] = (target >= threshold).astype(int)

    # Save the full preprocessed dataset.
    df.to_csv(args.all_train_output, sep="\t", index=False)

    # Split into training (90%) and test (10%)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Save training and test subsets.
    train_df.to_csv(args.train_output, sep="\t", index=False)
    test_df.to_csv(args.test_output, sep="\t", index=False)

    print("Data processing complete:")
    print(f"Full dataset shape: {df.shape}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print("Training target distribution:")
    print(train_df["Sleep_Quality_Score"].value_counts())

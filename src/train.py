import argparse
import pandas as pd
import numpy as np
import pickle
from decision_tree import learn_tree, print_tree, parse_criterion

def load_dataset(file_name):
    df = pd.read_csv(file_name, sep="\t")
    return df

def train_subtree(data, max_depth, criterion):
    criterion_func, optimize = criterion
    return learn_tree(data, max_depth, criterion_func, optimize)

def create_bootstrap_sample(df, seed):
    # Create a bootstrap sample from the full training data (sampling with replacement)
    return df.sample(n=len(df), replace=True, random_state=seed)

def write_subtrees_to_file(subtrees, text_file_name, pickle_file_name):
    # Write the text representation to a text file.
    # Open the text file in write mode to truncate it first.
    with open(text_file_name, "w") as f:
        for tree in subtrees:
            print_tree(tree, text_file_name)
    # Write the full forest as a pickle file (binary mode).
    with open(pickle_file_name, "wb") as f:
        pickle.dump(subtrees, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a random forest of 3 trees using bootstrap sampling"
    )
    # Use a single training file (created from your cleaned dataset)
    parser.add_argument("train_input", type=str,
                        help='Path to training input TSV file (e.g., data/train.tsv)')
    parser.add_argument("max_depth", type=int,
                        help='Maximum depth for each decision tree')
    parser.add_argument("criterion", type=str,
                        help='Splitting criterion (mutual_information, gini, or lowest_variance)')
    parser.add_argument("tree_text_out", type=str,
                        help='Path to output text file for the forest (e.g., train/forest.txt)')
    parser.add_argument("tree_pickle_out", type=str,
                        help='Path to output pickle file for the forest (e.g., train/forest.pkl)')
    args = parser.parse_args()

    # Parse the splitting criterion using your shared module.
    criterion = parse_criterion(args.criterion)

    # Load the training dataset.
    train_df = load_dataset(args.train_input)

    # Build a random forest of 3 trees using bootstrap sampling.
    subtrees = []
    for i in range(3):
        sample = create_bootstrap_sample(train_df, seed=42 + i)
        tree = train_subtree(sample, args.max_depth, criterion)
        subtrees.append(tree)

    # Write both the text version and the pickle file of the forest.
    write_subtrees_to_file(subtrees, args.tree_text_out, args.tree_pickle_out)

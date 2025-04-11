import argparse
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import decision_tree

def load_subtrees(file_name):
    """Load the list of trees (the forest) from a pickle file."""
    with open(file_name, "rb") as f:
        subtrees = pickle.load(f)
    return subtrees

def load_dataset(file_name):
    """Read a TSV file into a DataFrame."""
    return pd.read_csv(file_name, sep="\t")

def predict_row(row, node):
    if node.left is None and node.right is None:
        return node.vote
    # Compare the feature value against the threshold stored in the node
    if row[node.attr] <= node.threshold:
        return predict_row(row, node.left)
    else:
        return predict_row(row, node.right)


def predict_forest(row, subtrees):
    """Aggregate predictions from each subtree (using majority vote)."""
    votes = [predict_row(row, tree) for tree in subtrees]
    avg_vote = np.mean(votes)
    return 1 if avg_vote >= 0.5 else 0

def evaluate_forest(subtrees, df):
    """Evaluate the forest on a DataFrame. Returns accuracy and F1 score."""
    predictions = []
    actual = df.iloc[:, -1].tolist()
    for _, row in df.iterrows():
        pred = predict_forest(row, subtrees)
        predictions.append(pred)
    accuracy = accuracy_score(actual, predictions)
    f1 = f1_score(actual, predictions, average='macro')
    return accuracy, f1

def output_predictions(subtrees, df, out_file):
    """
    Write a file with the index, actual label, and predicted label for each instance.
    """
    with open(out_file, "w") as f:
        f.write("Index\tActual\tPredicted\n")
        for idx, row in df.iterrows():
            actual = row.iloc[-1]
            predicted = predict_forest(row, subtrees)
            f.write(f"{idx}\t{actual}\t{predicted}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tree_input", type=str, help='Path to file containing the pickled forest')
    parser.add_argument("train_input", type=str, help='Path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='Path to test input .tsv file')
    parser.add_argument("metrics_out", type=str, help='Path to output .txt file for evaluation metrics')
    parser.add_argument("predictions_out", type=str, help='Path to output .txt file for predictions vs actual')
    args = parser.parse_args()

    # load the saved forest (list of subtrees)
    subtrees = load_subtrees(args.tree_input)
    # load the datasets
    train_df = load_dataset(args.train_input)
    test_df = load_dataset(args.test_input)

    # evaluate on both training and testing data
    train_accuracy, train_f1 = evaluate_forest(subtrees, train_df)
    test_accuracy, test_f1 = evaluate_forest(subtrees, test_df)

    with open(args.metrics_out, "w") as f:
        f.write("Train Accuracy: {}\n".format(train_accuracy))
        f.write("Train F1 Score: {}\n".format(train_f1))
        f.write("Test Accuracy: {}\n".format(test_accuracy))
        f.write("Test F1 Score: {}\n".format(test_f1))

    output_predictions(subtrees, test_df, args.predictions_out)
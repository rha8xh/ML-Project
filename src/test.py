import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tree_input", type=str, help='path to tree')
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to test input .tsv file')
    parser.add_argument("metrics_out", type=str, help='where to write metrics to')
    args = parser.parse_args()

    # TODO: Evaluate tree on both train and test and report metrics (accuracy and F1)
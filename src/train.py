import argparse
import pandas as pd
import numpy as np
import pickle
import decision_tree

def load_dataset(file_name):
    df = pd.read_csv(file_name, sep="\t")
    return df

def train_subtree(data, max_depth, criterion):
    criterion_func, optimize = criterion
    return decision_tree.learn_tree(data, max_depth, criterion_func, optimize)

def write_subtrees_to_file(subtrees, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(subtrees, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input_1", type=str, help='Path to first training input .tsv file')
    parser.add_argument("train_input_2", type=str, help='Path to second training input .tsv file')
    parser.add_argument("train_input_3", type=str, help='Path to third training input .tsv file')
    parser.add_argument("train_input_4", type=str, help='Path to fourth training input .tsv file')
    parser.add_argument("max_depth", type=int, help='Maximum depth for the tree')
    parser.add_argument("criterion", type=str, help='Splitting criterion (mutual_information, gini, or lowest_variance)')
    parser.add_argument("tree_out", type=str, help='Path to output file where the forest is saved')
    args = parser.parse_args()

    # parse the splitting criterion using the shared module
    criterion = decision_tree.parse_criterion(args.criterion)

    # load training datasets
    data1 = load_dataset(args.train_input_1)
    data2 = load_dataset(args.train_input_2)
    data3 = load_dataset(args.train_input_3)
    data4 = load_dataset(args.train_input_4)

    # train one subtree on each training subset
    subtrees = []
    subtrees.append(train_subtree(data1, args.max_depth, criterion))
    subtrees.append(train_subtree(data2, args.max_depth, criterion))
    subtrees.append(train_subtree(data3, args.max_depth, criterion))
    subtrees.append(train_subtree(data4, args.max_depth, criterion))

    print("Printing the structure of the first subtree:")
    decision_tree.print_tree(subtrees[0])

    # save the forest to file
    write_subtrees_to_file(subtrees, args.tree_out)
import argparse


class Node:
    # TODO
    pass


def load_dataset(file_name):
    # TODO
    pass


def parse_criterion(criterion):
    # TODO
    pass


def train_subtree(data, labels, max_depth, criterion):
    # TODO
    pass


def write_subtrees_to_file(subtrees, file_name):
    # TODO
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input_1", type=str, help='path to first training input .tsv file')
    parser.add_argument("train_input_2", type=str, help='path to second training input .tsv file')
    parser.add_argument("train_input_3", type=str, help='path to third training input .tsv file')
    parser.add_argument("train_input_4", type=str, help='path to fourth training input .tsv file')
    parser.add_argument("max_depth", type=int,
                        help='maximum depth to which the tree should be built')
    parser.add_argument("criterion", type=str, help='splitting criterion used to train this tree')
    parser.add_argument("tree_out", type=str,
                        help='path of the output .txt file to which the tree should be written')
    args = parser.parse_args()

    criterion = parse_criterion(args.criterion)

    data_1, labels_1 = load_dataset(args.train_input_1)
    data_2, labels_2 = load_dataset(args.train_input_2)
    data_3, labels_3 = load_dataset(args.train_input_3)
    data_4, labels_4 = load_dataset(args.train_input_4)

    subtrees = []
    subtrees.add(train_subtree(data = data_1, labels = labels_1, max_depth = args.max_depth, criterion = criterion))
    subtrees.add(train_subtree(data = data_2, labels = labels_2, max_depth = args.max_depth, criterion = criterion))
    subtrees.add(train_subtree(data = data_3, labels = labels_3, max_depth = args.max_depth, criterion = criterion))
    subtrees.add(train_subtree(data = data_4, labels = labels_4, max_depth = args.max_depth, criterion = criterion))

    write_subtrees_to_file(subtrees, args.tree_out)

class Node:
    # TODO
    pass


def load_dataset(file_name):
    # TODO
    pass


def train_subtree(data, labels, max_depth):
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
    parser.add_argument("tree_out", type=str,
                        help='path of the output .txt file to which the tree should be written')
    args = parser.parse_args()

    train = []
    train.add(load_dataset(args.train_input_1))
    train.add(load_dataset(args.train_input_2))
    train.add(load_dataset(args.train_input_3))
    train.add(load_dataset(args.train_input_4))

    write_subtrees_to_file(train, args.tree_out)

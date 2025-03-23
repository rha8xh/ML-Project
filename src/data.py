import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_input", type=str, help='path to whole datset')
    parser.add_argument("train_output_1", type=str, help='path to first training input .tsv file')
    parser.add_argument("train_output_2", type=str, help='path to second training input .tsv file')
    parser.add_argument("train_output_3", type=str, help='path to third training input .tsv file')
    parser.add_argument("train_output_4", type=str, help='path to fourth training input .tsv file')
    parser.add_argument("test_output", type=str, help='path to test input .tsv file')
    args = parser.parse_args()

    # TODO: Remove unwanted columns, reserve ~10% for testing, randomly divide rest into four subgroups
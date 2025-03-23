import argparse
import math
import pandas as pd


class Node:
    def __init__(self, attr, attr_val, depth, df):
        self.left = None
        self.right = None
        self.attr = attr
        self.attr_val = attr_val
        self.depth = depth
        self.df = df
        self.counts = self.get_counts()
        self.vote = self.get_vote()    
        
    def get_counts(self):
        return [(self.df.iloc[:, -1] == 0).sum(), (self.df.iloc[:, -1] == 1).sum()]
        
    def get_vote(self):
        if (self.counts[0] > self.counts[1]): return 0
        else: return 1
        
    def split(self):
        attr = self.get_max_info_attr()
        if (attr == None):
            return None, None, None
        df_0 = self.df[self.df[attr] == 0]
        df_1 = self.df[self.df[attr] == 1]
        return attr, df_0, df_1
        
    def get_max_info_attr(self):
        Y = self.df.iloc[:, -1]
        max_info_attr = None
        max_info = 0
        for (attr, X) in self.df.iloc[:, :-1].items():
            info = mutual_information(X.tolist(), Y.tolist())
            if (info > max_info):
                max_info = info
                max_info_attr = attr
        return max_info_attr
        
        
def learn_tree(train, max_depth):
    tree = Node("ROOT", None, 0, train)
    learn_node(tree, max_depth)
    return tree
    
    
def learn_node(node, max_depth):
    if (node.depth >= max_depth):
        return
    attr, df_0, df_1 = node.split()
    if (attr == None): return
    node.left = Node(attr, 0, node.depth + 1, df_0)
    node.right = Node(attr, 1, node.depth + 1, df_1)
    learn_node(node.left, max_depth)
    learn_node(node.right, max_depth)


def print_tree(tree, file):
    with open(file, "w") as file:
        print_tree_rec(tree, file)


def print_tree_rec(node, file):
    if (node.depth == 0):
        file.write("[%d 0/%d 1]\n" % (node.counts[0], node.counts[1]))
    else:
        file.write("| " * node.depth)
        file.write("%s = %d: [%d 0/%d 1]\n" % (node.attr, node.attr_val, node.counts[0], node.counts[1]))
    if (node.left != None):
        print_tree_rec(node.left, file)
    if (node.right != None):
            print_tree_rec(node.right, file)
        
        
def predict_data(df, tree, file):
    num_errors = 0
    with open(file, "w") as file:
        for row in df.itertuples():
            prediction = predict_row(row, tree)
            file.write("%d\n" % prediction)
            if (prediction != row[-1]):
                num_errors += 1
    return num_errors / len(df.index)
            

def predict_row(row, node):
    if (node.left == None and node.right == None):
        return node.vote
    attr = node.left.attr
    if (getattr(row, attr) == node.left.attr_val):
        return predict_row(row, node.left)
    else:
        return predict_row(row, node.right)
        
        
def print_metrics(train_error, test_error, file):
    with open(file, "w") as file:
        file.write("error(train): %f\n" % train_error)
        file.write("error(test): %f\n" % test_error)
    
    
def mutual_information(X, Y):
    ent_Xis0 = None
    num_0_in_X = X.count(0)
    if (num_0_in_X > 0):
        ent_Xis0 = entropy([y for x, y in zip(X, Y) if x == 0])
        ent_Xis0 *= (num_0_in_X / len(X))
    else:
        ent_Xis0 = 0
        
    ent_X_is1 = None
    num_1_in_X = X.count(1)
    if (num_1_in_X > 0):
        ent_Xis1 = entropy([y for x, y in zip(X, Y) if x == 1])
        ent_Xis1 *= (num_1_in_X / len(X))
    else:
        ent_Xis1 = 0
        
    return entropy(Y) - ent_Xis0 - ent_Xis1
    
    
def entropy(array):
    num_0 = array.count(0)
    num_1 = array.count(1)
    if (num_0 == 0 or num_1 == 0): return 0
    p_0 = num_0 / len(array)
    p_1 = num_1 / len(array)
    return -1 * (p_0 * math.log(p_0, 2) + p_1 * math.log(p_1, 2))
    

def parseData(file):
    df = pd.read_csv(file, sep = "\t")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    train = parseData(args.train_input)
    test = parseData(args.test_input)
    tree = learn_tree(train = train, max_depth = args.max_depth) 
    print_tree(tree = tree, file = args.print_out)
    train_error = predict_data(df = train, tree = tree, file = args.train_out)
    test_error = predict_data(df = test, tree = tree, file = args.test_out)
    print_metrics(train_error = train_error, test_error = test_error, file = args.metrics_out)

import math
import pandas as pd

def mutual_information(X, Y):
    ent_Y = entropy(Y)
    num_0_in_X = X.count(0)
    if num_0_in_X > 0:
        ent_Xis0 = entropy([y for x, y in zip(X, Y) if x == 0]) * (num_0_in_X / len(X))
    else:
        ent_Xis0 = 0
    num_1_in_X = X.count(1)
    if num_1_in_X > 0:
        ent_Xis1 = entropy([y for x, y in zip(X, Y) if x == 1]) * (num_1_in_X / len(X))
    else:
        ent_Xis1 = 0
    return ent_Y - ent_Xis0 - ent_Xis1

def entropy(array):
    num_0 = array.count(0)
    num_1 = array.count(1)
    if num_0 == 0 or num_1 == 0:
        return 0
    p_0 = num_0 / len(array)
    p_1 = num_1 / len(array)
    return - (p_0 * math.log(p_0, 2) + p_1 * math.log(p_1, 2))

def gini_index(X, Y):
    total = len(Y)
    indices0 = [i for i, x in enumerate(X) if x == 0]
    indices1 = [i for i, x in enumerate(X) if x == 1]
    
    def impurity(indices):
        if len(indices) == 0:
            return 0
        subset = [Y[i] for i in indices]
        p0 = subset.count(0) / len(subset)
        p1 = subset.count(1) / len(subset)
        return 1 - (p0**2 + p1**2)
    
    weighted_impurity = (len(indices0)/total) * impurity(indices0) + (len(indices1)/total) * impurity(indices1)
    return weighted_impurity

def lowest_variance(X, Y):
    p = X.count(1) / len(X)
    return p * (1 - p)

def parse_criterion(criterion):
    """
    Given a string, return a tuple: (criterion_function, optimization_direction).
    For 'mutual_information' we maximize; for 'gini' or 'lowest_variance' we minimize.
    """
    criterion = criterion.lower()
    if criterion == 'mutual_information':
        return mutual_information, 'max'
    elif criterion == 'gini':
        return gini_index, 'min'
    elif criterion == 'lowest_variance':
        return lowest_variance, 'min'
    else:
        raise ValueError("Unknown criterion: " + criterion)

class Node:
    def __init__(self, attr=None, threshold=None, depth=0, df=None, criterion_func=mutual_information, optimize='max'):
        self.left = None
        self.right = None
        self.attr = attr # the attribute used for splitting
        self.threshold = threshold # the attribute value leading to this node
        self.compare_symbol = None
        self.depth = depth
        self.df = df # the df at this node
        self.criterion_func = criterion_func
        self.optimize = optimize
        self.counts = self.get_counts() # [# of 0 labels, # of 1 labels]
        self.vote = self.get_vote() # majority vote for the node
        self.chosen_attr = list()

    def get_counts(self):
        print(self.df.shape)
        return [(self.df.iloc[:, -1] == 0).sum(), (self.df.iloc[:, -1] == 1).sum()]

    def get_vote(self):
        return 0 if self.counts[0] > self.counts[1] else 1

    def get_best_attribute(self):
        func = self.criterion_func
        optimize = self.optimize
        Y = self.df.iloc[:, -1].tolist()
        best_attr = None
        best_score = None
        for (attr, X) in self.df.iloc[:, :-1].items():
            if (attr in self.chosen_attr):
                continue
            score = func(X.tolist(), Y)
            print(f"Attribute: {attr}, Score: {score}")
            if best_score is None:
                best_score = score
                best_attr = attr
            else:
                if optimize == 'max' and score > best_score:
                    best_score = score
                    best_attr = attr
                elif optimize == 'min' and score < best_score:
                    best_score = score
                    best_attr = attr
        print("Chosen attribute:", best_attr)
        return best_attr

    def best_threshold(self, attr):
        # Get sorted unique values for the attribute
        unique_vals = sorted(self.df[attr].unique())
        best_thresh = None
        best_score = float('inf')
        # Try candidate thresholds between each pair of unique values
        for i in range(len(unique_vals) - 1):
            # Candidate threshold is the midpoint between consecutive values
            thresh = (unique_vals[i] + unique_vals[i+1]) / 2.0
            df_left = self.df[self.df[attr] <= thresh]
            df_right = self.df[self.df[attr] > thresh]
            # Skip candidate thresholds that result in an empty split
            if df_left.empty or df_right.empty:
                continue
            # Compute a splitting score (for instance, weighted Gini impurity)
            score = self.compute_split_score(df_left, df_right)
            if score < best_score:
                best_score = score
                best_thresh = thresh
        return best_thresh, best_score

    def compute_split_score(self, df_left, df_right):
        # Here you can compute the weighted impurity of the split.
        # This is just an example using Gini impurity.
        def gini(df):
            target = df.iloc[:, -1].tolist()
            if len(target) == 0:
                return 0
            p = target.count(1) / len(target)
            return 1 - p**2 - (1 - p)**2
        n = len(self.df)
        score = (len(df_left)/n) * gini(df_left) + (len(df_right)/n) * gini(df_right)
        return score

    # def split(self):
    #     attr = self.get_best_attribute()
    #     if attr is None:
    #         return None, None, None
    #     df_0 = self.df[self.df[attr] == 0]
    #     df_1 = self.df[self.df[attr] == 1]
    #     return attr, df_0, df_1
    def split(self):
        # First, choose the best attribute.
        attr = self.get_best_attribute()
        if attr is None:
            return None  # No attribute available

        # Now, search for the best threshold for splitting on this attribute.
        best_thresh, score = self.best_threshold(attr)
        if best_thresh is None:
            print(f"No valid split found for {attr}. Node becomes a leaf.")
            return None  # Signal that a valid split was not found

        # self.threshold = best_thresh  # store the found threshold
        print(f"Splitting on {attr} with threshold {best_thresh}")

        # Partition the data using the threshold.
        df_left = self.df[self.df[attr] <= best_thresh]
        df_right = self.df[self.df[attr] > best_thresh]
        if df_left.empty or df_right.empty:
            print(f"Empty split encountered for {attr} at threshold {best_thresh}. Node becomes a leaf.")
            return None

        return attr, best_thresh, df_left, df_right



def learn_tree(df, max_depth, criterion_func, optimize):
    root = Node(None, None, 0, df, criterion_func, optimize)
    learn_node(root, max_depth)
    return root

# def learn_node(node, max_depth):
#     if node.depth >= max_depth:
#         print("Max depth reached at node with counts:", node.counts)
#         return
    
#     # Use the custom split() method
#     attr, df_left, df_right = node.split()
#     if df_left is None or df_right is None or df_left.empty or df_right.empty:
#         print("Empty split encountered at node with counts:", node.counts)
#         return
    
#     print(f"Splitting on attribute {attr} at depth {node.depth}")
#     node.left = Node(attr, 0, node.depth + 1, df_left, node.criterion_func, node.optimize)
#     node.right = Node(attr, 1, node.depth + 1, df_right, node.criterion_func, node.optimize)
#     learn_node(node.left, max_depth)
#     learn_node(node.right, max_depth)
def learn_node(node, max_depth):
    if node.depth >= max_depth:
        print("Max depth reached at node with counts:", node.counts)
        return

    # Try to split the node.
    result = node.split()
    if result is None:
        # No valid split was found; this node will remain a leaf.
        print("No valid split; node becomes a leaf with counts:", node.counts)
        return

    attr, threshold, df_left, df_right = result

    # Update the node's splitting attribute (it will be valid now).
    # node.attr = attr

    # Create left and right child nodes using the split data.
    node.left = Node(attr, threshold, node.depth + 1, df_left, node.criterion_func, node.optimize)
    node.right = Node(attr, threshold, node.depth + 1, df_right, node.criterion_func, node.optimize)
    node.left.compare_symbol = "<"
    node.right.compare_symbol = ">"
    node.chosen_attr.append(attr)
    node.left.chosen_attr.extend(node.chosen_attr)
    node.right.chosen_attr.extend(node.chosen_attr)
    learn_node(node.left, max_depth)
    learn_node(node.right, max_depth)


"""
def print_tree(node, indent=""):
    if node is None:
        return
    print(indent + f"Node (attr: {node.attr}, vote: {node.vote}, counts: {node.counts})")
    if node.left is not None or node.right is not None:
        print_tree(node.left, indent + "  ")
        print_tree(node.right, indent + "  ")
"""
def print_tree(tree, file):
    with open(file, "a") as file:
        print_tree_rec(tree, file)


def print_tree_rec(node, file):
    if (node.depth == 0):
        file.write("[%d 0/%d 1]\n" % (node.counts[0], node.counts[1]))
    else:
        file.write("| " * node.depth)
        file.write("%s %s %.1f: [%d 0/ %d 1]\n" % (node.attr, node.compare_symbol, node.threshold, node.counts[0], node.counts[1]))
    if (node.left != None):
        print_tree_rec(node.left, file)
    if (node.right != None):
        print_tree_rec(node.right, file)
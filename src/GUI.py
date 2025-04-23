import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from decision_tree import learn_tree, parse_criterion
import networkx as nx


class SleepQualityApp:
    def __init__(self, root):
        self.root = root
        root.title("Sleep Quality Predictor")

        self.df = None
        self.tree = None

        input_frame = ttk.LabelFrame(root, text="Input Features")
        input_frame.pack(padx=10, pady=5, fill="x")

        self.entries = {}
        self.create_input(input_frame, "Gender (0=F, 1=M):", "gender")
        self.create_input(input_frame, "Age:", "age")
        self.create_input(input_frame, "Sleep Duration (hrs):", "sleep_duration")
        self.create_input(input_frame, "Physical Activity (0-10):", "physical_activity")
        self.create_input(input_frame, "BMI:", "bmi")
        self.create_input(input_frame, "Stress Level (0-40):", "stress_level")

        ttk.Label(input_frame, text="Splitting Criterion:").pack()
        self.criterion_var = tk.StringVar(value="gini")
        ttk.Combobox(input_frame, textvariable=self.criterion_var,
                     values=["gini", "mutual_information", "lowest_variance"], state='readonly').pack()

        self.depth_var = tk.IntVar(value=3)
        self.create_input(input_frame, "Max Tree Depth:", "depth", var=self.depth_var)

        control_frame = ttk.Frame(root)
        control_frame.pack(pady=5)
        ttk.Button(control_frame, text="Upload CSV", command=self.upload_csv).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Train Tree", command=self.train_tree).pack(side="left", padx=5)
        self.visualize_button = ttk.Button(root, text="Visualize Tree", command=self.visualize_tree)
        self.visualize_button.pack(pady=5)

        ttk.Button(control_frame, text="Predict", command=self.predict).pack(side="left", padx=5)

        self.result_label = ttk.Label(root, text="")
        self.result_label.pack()


    def visualize_tree(self):
        if self.tree is None:
            self.result_label.config(text="Train the tree first.")
            return

        G = nx.DiGraph()
        pos = {}
        labels = {}

        def traverse(node, parent=None, direction='', depth=0, pos_x=0):
            if node is None:
                return

            node_id = id(node)
            label = f"{node.attr}\n{node.compare_symbol} {node.threshold:.1f}" if node.attr else f"Leaf\nVote: {node.vote}"
            G.add_node(node_id)
            labels[node_id] = label
            pos[node_id] = (pos_x, -depth)

            if parent is not None:
                G.add_edge(parent, node_id, label=direction)

            if node.left:
                traverse(node.left, node_id, '<=', depth + 1, pos_x - 2 ** (3 - depth))
            if node.right:
                traverse(node.right, node_id, '>', depth + 1, pos_x + 2 ** (3 - depth))

        traverse(self.tree)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, arrows=True, node_size=2000, node_color='skyblue')
        nx.draw_networkx_labels(G, pos, labels)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Decision Tree Visualization")
        plt.tight_layout()
        plt.show()

    def create_input(self, parent, label, key, var=None):
        ttk.Label(parent, text=label).pack()
        entry_var = var or tk.StringVar()
        entry = ttk.Entry(parent, textvariable=entry_var)
        entry.pack()
        self.entries[key] = entry_var

    def upload_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.df = pd.read_csv(path)

            label_column = self.df.columns[-1]
            median_value = self.df[label_column].median()
            self.df["Label"] = (self.df[label_column] > median_value).astype(int)
            self.df = self.df.drop(columns=[label_column])

            self.result_label.config(text=f"Loaded {len(self.df)} rows. Labels binarized.")

    def train_tree(self):
        if self.df is None:
            self.result_label.config(text="Please upload a CSV first.")
            return

        if "Label" not in self.df.columns:
            self.result_label.config(text="Last column should be labeled as 'Label'.")
            return

        criterion_func, optimize = parse_criterion(self.criterion_var.get())
        self.tree = learn_tree(self.df, self.depth_var.get(), criterion_func, optimize)
        self.result_label.config(text="Tree trained successfully!")

    def predict(self):
        try:
            if self.tree is None:
                self.result_label.config(text="Train the tree first.")
                return

            input_data = {
                'Gender': int(self.entries['gender'].get()),
                'Age': float(self.entries['age'].get()),
                'Sleep Duration': float(self.entries['sleep_duration'].get()),
                'Physical Activity': float(self.entries['physical_activity'].get()),
                'BMI': float(self.entries['bmi'].get()),
                'Stress Level': float(self.entries['stress_level'].get())
            }
            df_input = pd.DataFrame([input_data])

            node = self.tree
            while node.left is not None and node.right is not None:
                value = df_input.iloc[0][node.attr]
                if value <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            prediction = node.vote if node is not None else "Unknown"
            self.result_label.config(text=f"Predicted Sleep Quality: {prediction}")

        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SleepQualityApp(root)
    root.mainloop()

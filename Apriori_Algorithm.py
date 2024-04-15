import tkinter as tk
from collections import Counter
from itertools import combinations
from tkinter import filedialog
import pandas as pd
from tabulate import tabulate


class AprioriWithGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Apriori Algorithm GUI")
        self.root.configure(bg="#F5F5DC")

        # GUI elements
        tk.Label(root, text="Min Support Count:").pack()
        self.min_support_count_entry = tk.Entry(root)
        self.min_support_count_entry.pack()

        tk.Label(root, text="Min Confidence:").pack()
        self.confidence_entry = tk.Entry(root)
        self.confidence_entry.pack()

        tk.Label(root, text="Percentage of Data (%):").pack()
        self.percentage_entry = tk.Entry(root)
        self.percentage_entry.pack()

        self.preprocess_button = tk.Button(root, text="Choose File", command=self.preprocess_and_find_association_rules)
        self.preprocess_button.pack()

        # Text widget to display association rules
        self.frequent_item_sets_text = tk.Text(root, height=10, width=50)
        self.frequent_item_sets_text.pack()

        # Text widget to display association rules
        self.association_rules_text = tk.Text(root, height=10, width=50)
        self.association_rules_text.pack()

    def preprocess_and_find_association_rules(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        min_support_count = int(self.min_support_count_entry.get())
        min_confidence = float(self.confidence_entry.get())
        percentage = float(self.percentage_entry.get())

        df = pd.read_csv(file_path)
        num_records = int(len(df) * (percentage / 100))
        df = df.head(num_records)

        # Preprocess data
        data = self.preprocess_data(df)

        # Find frequent item sets
        frequent_item_sets = self.find_frequent_item_sets(data, min_support_count)

        if frequent_item_sets:
            # Display frequent item sets
            self.display_frequent_itemset(frequent_item_sets)
            # Generate association rules
            association_rules = self.generate_association_rules(frequent_item_sets, min_confidence)
            print("Association Rules generated :", association_rules)  # Add this line for debugging

        # Display association rules
        self.display_association_rules(association_rules)


    def preprocess_data(self, df):
        # Check for missing values
        missing_values = df.isnull().sum()
        print('Missing values:\n', missing_values)
        print('----------------------')

        # Remove unused columns
        columns_to_remove = ['DateTime', 'Daypart', 'DayType']
        data = df.drop(columns=columns_to_remove)

        # Aggregate transactions
        aggregated_data = data.groupby('TransactionNo')['Items'].agg(list).reset_index()
        print((aggregated_data))
        return aggregated_data

    def find_frequent_item_sets(self, data, min_support_count):
        all_items = [item for sublist in data['Items'] for item in sublist]
        unique_items = set(all_items)

        # Initialize k to 1 for the first iteration
        k = 1
        Lk_1 = []
        frequent_item_sets = {}
        while True:
            print("Processing iteration", k)

            # Generate candidate itemsets Ck
            if k == 1:
                Ck_items = {frozenset([item]) for item in unique_items}
            else:
                Ck_items = self.generate_candidate_itemsets(Lk_1, k)

            # Get frequent itemsets Lk
            Lk = self.generate_frequent_itemsets(data, Ck_items, min_support_count)
            print(tabulate([[', '.join(list(item)), support_count] for item, support_count in Lk.items()],
                           headers=["Item", "Support Count"]))

            if not Lk:
                print("No frequent itemsets found for iteration", k)
                break

            # Prepare for the next iteration
            k += 1
            Lk_1 = set(Lk.keys())
            frequent_item_sets.update(Lk)

        print("Apriori algorithm completed")
        return frequent_item_sets

    def generate_candidate_itemsets(self, Lk_1, k):
        Ck_items = set()
        for itemset1 in Lk_1:
            for itemset2 in Lk_1:
                if len(itemset1.union(itemset2)) == k:
                    Ck_items.add(itemset1.union(itemset2))
        return Ck_items

    def generate_frequent_itemsets(self, data, Ck_items, min_support_count):
        Ck_support_counts = Counter()
        for items in data['Items']:
            for itemset in Ck_items:
                if all(item in items for item in itemset):
                    Ck_support_counts[itemset] += 1

        Lk = {itemset: support_count for itemset, support_count in Ck_support_counts.items() if
              support_count >= min_support_count}
        return Lk

    def generate_association_rules(self, frequent_item_sets, min_confidence):
        association_rules = []
        for itemset in frequent_item_sets.keys():
            if len(itemset) > 1:
                self.generate_rules_from_itemset(itemset, frequent_item_sets, association_rules, min_confidence)
        print("Association Rules:", association_rules)  # Add this line for debugging
        return association_rules

    def generate_rules_from_itemset(self, itemset, frequent_item_sets, association_rules, min_confidence):
        for i in range(len(itemset) - 1, 0, -1):  # Start from the largest combination
            for LHS in combinations(itemset, i):
                LHS_set = frozenset(LHS)
                confidence = frequent_item_sets[itemset] / frequent_item_sets[LHS_set]
                if confidence >= min_confidence:
                    RHS = itemset.difference(LHS_set)
                    association_rules.append((LHS, RHS, confidence))

    def display_frequent_itemset(self, frequent_item_sets):
        self.frequent_item_sets_text.delete(1.0, tk.END)  # Clear previous content
        self.frequent_item_sets_text.insert(tk.END, "Frequent Item Sets:\n")

        # Display each item set
        for item_set in frequent_item_sets:
            items_str = ", ".join(item_set)
            support_count = frequent_item_sets[item_set]
            self.frequent_item_sets_text.insert(tk.END, f"Item Set: {{{items_str}}}, Support Count: {support_count}\n")

        # Expand the text widget vertically and horizontally
        self.frequent_item_sets_text.pack(fill=tk.BOTH, expand=True)



    def display_association_rules(self, association_rules):
        self.association_rules_text.delete(1.0, tk.END)  # Clear previous content
        self.association_rules_text.insert(tk.END, "Strong Association Rules:\n")
        for rule in association_rules:
            LHS = "{" + ', '.join(rule[0]) + "}"
            RHS = "{" + ', '.join(rule[1]) + "}"
            self.association_rules_text.insert(tk.END, f"Rule: {LHS} => {RHS}, Confidence: {rule[2]}\n")
            # Expand the text widget vertically and horizontally
            self.association_rules_text.pack(fill=tk.BOTH, expand=True)



if __name__ == "__main__":
    root = tk.Tk()
    app = AprioriGUI(root)
    # Set window size and allow it to expand
    root.resizable(True, True)  # Allow both horizontal and vertical resizing
    root.mainloop()

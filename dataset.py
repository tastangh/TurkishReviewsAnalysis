import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import torch


class DataProcessor:
    def __init__(self, dataset_name, text_column='text', label_column='label', random_state=42):
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state
        self.data = self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset(self.dataset_name)
        return dataset['train'].to_pandas()

    def get_random_subset(self, subset_size=5000):
        subset = self.data.sample(n=subset_size, random_state=self.random_state)
        return subset

    def split_data(self, subset, test_size=0.2):
        X = subset[self.text_column]
        y = subset[self.label_column]
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    @staticmethod
    def plot_label_distribution(data, label_column, title='Label Distribution', save_dir='plots'):
        # Create a bar plot for label distribution
        label_counts = data[label_column].value_counts().reset_index()
        label_counts.columns = [label_column, 'counts']

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=label_counts, 
            x=label_column, 
            y='counts', 
            hue=label_column,  
            dodge=False, 
            legend=False, 
            palette='viridis'
        )

        plt.title(title)
        plt.xlabel('Sınıflar')
        plt.ylabel('Eleman Sayısı')

        for i, count in enumerate(label_counts['counts']):
            plt.text(i, count + 5, f"{count}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(filename, dpi=300)
        print(f"[INFO] Plot saved at {filename}")

        # plt.show()
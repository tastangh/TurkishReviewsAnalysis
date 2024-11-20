from datasets import load_dataset
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

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

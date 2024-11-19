from datasets import load_dataset
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, dataset_name, text_column='text', label_column='label', random_state=42):
        """
        DataProcessor sınıfı, Hugging Face'den veri setini indirir ve işlemeye hazırlar.
        
        Args:
            dataset_name -- Hugging Face veri setinin adı
            text_column -- Metin sütununun adı (default: 'text')
            label_column -- Etiket sütununun adı (default: 'label')
            random_state -- Rastgele seçimler için sabit bir seed değeri
        """
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state
        self.data = self.load_dataset()

    def load_dataset(self):
        """
        Hugging Face veri setini indirir.
        
        Returns:
            Pandas DataFrame -- indirilen veri seti
        """
        # Hugging Face veri setini indir
        dataset = load_dataset(self.dataset_name)
        # Hugging Face DatasetDict'i Pandas DataFrame'e dönüştür
        return dataset['train'].to_pandas()

    def get_random_subset(self, subset_size=5000):
        """
        Veri setinden rastgele bir alt küme seçer.
        
        Args:
            subset_size -- Alt kümenin büyüklüğü (default: 5000)
        
        Returns:
            Pandas DataFrame -- Alt küme veri seti
        """
        # Rastgele bir alt küme seçimi
        subset = self.data.sample(n=subset_size, random_state=self.random_state)
        return subset

    def split_data(self, subset, test_size=0.2):
        """
        Alt kümeyi eğitim ve test seti olarak böler.
        
        Args:
            subset -- Veri alt kümesi
            test_size -- Test seti oranı (default: 0.2)
        
        Returns:
            X_train, X_test, y_train, y_test -- Eğitim ve test setleri
        """
        X = subset[self.text_column]  # Dinamik olarak belirtilen metin sütunu
        y = subset[self.label_column]  # Dinamik olarak belirtilen etiket sütunu
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

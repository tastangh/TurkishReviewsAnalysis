import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, dataset_url, random_state=42):
        """
        DataProcessor sınıfı, Hugging Face'den veri setini indirir ve işlemeye hazırlar.
        
        Args:
            dataset_url -- Hugging Face veri setinin indirilme URL'si
            random_state -- Rastgele seçimler için sabit bir seed değeri
        """
        self.dataset_url = dataset_url
        self.random_state = random_state
        self.data = self.load_dataset()

    def load_dataset(self):
        """
        Hugging Face URL'sinden veri setini indirir.
        
        Returns:
            Pandas DataFrame -- indirilen veri seti
        """
        # Veri setini Hugging Face'den indir
        data = pd.read_json(self.dataset_url, lines=True)
        return data

    def get_random_subset(self, subset_size=5000):
        """
        Veri setinden rastgele ancak aynı random_state ile bir alt küme seçer.
        
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
        X = subset['text']  # Metin sütunu
        y = subset['label']  # Etiket sütunu
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


class DataProcessor:
    """
    Hugging Face veri setlerini yüklemek, işlemek, eğitim/test ayırımı yapmak ve görselleştirmek için bir sınıf.

    Attributes:
        dataset_name (str): Yüklenmek istenen veri setinin adı (ör. "imdb").
        text_column (str): Metin verilerini barındıran sütunun adı (varsayılan: "text").
        label_column (str): Etiketleri barındıran sütunun adı (varsayılan: "label").
        random_state (int): Rastgelelik için kullanılan sabit değer (varsayılan: 42).
        data (pd.DataFrame): Veri setinin pandas DataFrame formatındaki hali.
    """

    def __init__(self, dataset_name, text_column='text', label_column='label', random_state=42):
        """
        DataProcessor sınıfının yapıcı metodu.

        Args:
            dataset_name (str): Yüklenmek istenen veri setinin adı.
            text_column (str): Metin verilerini içeren sütunun adı.
            label_column (str): Etiketleri içeren sütunun adı.
            random_state (int): Rastgelelik için kullanılan sabit değer.
        """
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.label_column = label_column
        self.random_state = random_state
        self.data = self.load_dataset()

    def load_dataset(self):
        """
        Hugging Face üzerinden veri setini yükler.

        Returns:
            pd.DataFrame: Yüklenen veri setinin pandas DataFrame hali.
        Raises:
            ValueError: Veri seti yüklenemezse bir hata fırlatır.
        """
        try:
            print(f"[INFO] Veri seti yüklenmeye çalışılıyor: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)  # Hugging Face veri setini yükle
        except ValueError as e:
            print(f"[ERROR] Hugging Face'ten veri seti yüklenemedi: {e}")
            raise ValueError(f"Veri seti '{self.dataset_name}' yüklenemedi.")
        except Exception as e:
            print(f"[ERROR] Beklenmeyen bir hata oluştu: {e}")
            raise
        return dataset['train'].to_pandas()  # 'train' kısmını pandas formatına çevir

    def get_random_subset(self, subset_size=5000):
        """
        Veri setinden rastgele bir alt küme alır.

        Args:
            subset_size (int): Alt küme boyutu (varsayılan: 5000).

        Returns:
            pd.DataFrame: Rastgele seçilmiş alt küme.
        """
        subset = self.data.sample(n=subset_size, random_state=self.random_state)
        return subset

    def split_data(self, subset, test_size=0.2):
        """
        Eğitim ve test veri setlerini ayrıştırır.

        Args:
            subset (pd.DataFrame): Ayrıştırılacak veri kümesi.
            test_size (float): Test veri setinin oranı (varsayılan: %20).

        Returns:
            tuple: Eğitim ve test verileri (X_train, X_test, y_train, y_test).
        """
        X = subset[self.text_column]
        y = subset[self.label_column]
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    @staticmethod
    def plot_label_distribution(data, label_column, title='Label Distribution', save_dir='plots'):
        """
        Etiket dağılımını görselleştirir ve dosya olarak kaydeder.

        Args:
            data (pd.DataFrame): Etiket dağılımı görselleştirilecek veri kümesi.
            label_column (str): Etiketleri içeren sütunun adı.
            title (str): Grafik başlığı (varsayılan: "Label Distribution").
            save_dir (str): Grafiğin kaydedileceği klasörün adı (varsayılan: "plots").

        Saves:
            Grafik görüntüsü belirtilen klasöre kaydedilir.
        """
        # Etiketlerin sayısını bul
        label_counts = data[label_column].value_counts().reset_index()
        label_counts.columns = [label_column, 'counts']

        # Bar grafiği oluştur
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

        # Grafik başlığı ve eksen etiketleri
        plt.title(title)
        plt.xlabel('Sınıflar')
        plt.ylabel('Eleman Sayısı')

        # Barların üzerine sayıları ekle
        for i, count in enumerate(label_counts['counts']):
            plt.text(i, count + 5, f"{count}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Kaydetme işlemi
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(filename, dpi=300)
        print(f"[INFO] Grafik kaydedildi: {filename}")

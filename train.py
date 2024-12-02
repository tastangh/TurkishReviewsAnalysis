import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset import DataProcessor  # Veri işleme sınıfı
from embedding_generator import EmbeddingGenerator  # Embedding oluşturma sınıfı
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import torch_directml  # GPU/DirectML desteği için
import multiprocessing  # Paralel işlem yönetimi
import torch


class Trainer:
    """
    Makine öğrenmesi modellerini eğitmek ve değerlendirmek için bir sınıf.

    Attributes:
        X_train (np.ndarray): Eğitim veri özellikleri (embedding'ler).
        y_train (np.ndarray): Eğitim veri etiketleri.
        device (str): Eğitim için kullanılacak cihaz (ör. 'cuda' veya 'cpu').
        random_state (int): Rastgelelik kontrolü için sabit değer.
        models (dict): Eğitilen modelleri saklar.
        results (dict): Eğitim sonuçlarını (ör. doğruluk) saklar.
    """

    def __init__(self, X_train, y_train, device, random_state=42):
        """
        Trainer sınıfını başlatır.

        Args:
            X_train (np.ndarray): Eğitim veri özellikleri (embedding'ler).
            y_train (np.ndarray): Eğitim veri etiketleri.
            device (str): Eğitim için kullanılacak cihaz.
            random_state (int): Rastgelelik kontrolü için sabit değer.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.device = device
        self.random_state = random_state
        self.models = {}
        self.results = {}  # Eğitim sonuçlarını saklar

    def _ensure_dir_exists(self, path):
        """
        Belirtilen klasörün varlığını kontrol eder; yoksa oluşturur.

        Args:
            path (str): Oluşturulacak veya kontrol edilecek klasör yolu.
        """
        os.makedirs(path, exist_ok=True)

    def _plot_grid_search(self, grid, model_name, embedding_name):
        """
        GridSearch sonuçlarını çubuk grafiği ile görselleştirir.

        Args:
            grid (GridSearchCV): GridSearchCV sonucu.
            model_name (str): Modelin adı (ör. 'Random Forest').
            embedding_name (str): Embedding modelinin adı.
        """
        results = grid.cv_results_
        mean_scores = results['mean_test_score']
        params = results['params']

        sorted_indices = sorted(range(len(mean_scores)), key=lambda i: mean_scores[i], reverse=True)
        sorted_scores = [mean_scores[i] for i in sorted_indices]
        sorted_params = [params[i] for i in sorted_indices]

        plt.figure(figsize=(12, 6))
        barplot = sns.barplot(x=sorted_scores, y=[str(p) for p in sorted_params], palette="viridis")
        plt.title(f"{model_name} (Embedding: {embedding_name}) - Grid Search Results")
        plt.xlabel("Ortalama Test Doğruluğu")
        plt.ylabel("Hiperparametreler")
        plt.xlim(0, 1)

        for i, bar in enumerate(barplot.patches):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            plt.text(x + 0.01, y, f"%{x * 100:.3f}", ha='left', va='center', fontsize=10, color='black')

        plt.tight_layout()

        sanitized_embedding_name = embedding_name.replace("/", "_")
        sanitized_model_name = model_name.lower().replace(" ", "_")
        plot_dir = f"train_results/{sanitized_embedding_name}"
        self._ensure_dir_exists(plot_dir)

        plot_path = f"{plot_dir}/{sanitized_model_name}_grid_search.png"
        plt.savefig(plot_path)
        print(f"[SUCCESS] Grid Search görselleştirildi ve kaydedildi: {plot_path}")

    def train_rf(self, param_grid, embedding_name):
        """
        Random Forest modelini GridSearch ile eğitir.

        Args:
            param_grid (dict): Hiperparametre grid'i.
            embedding_name (str): Embedding modelinin adı.
        """
        print("\n[INFO] Random Forest GridSearchCV başlatılıyor...")
        rf = RandomForestClassifier(random_state=self.random_state)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=0)

        try:
            grid.fit(self.X_train, self.y_train)
            self.models['Random Forest'] = grid.best_estimator_
            self.results['Random Forest'] = grid.best_score_

            print(f"[SUCCESS] En iyi parametreler (RF): {grid.best_params_}")
            print(f"[INFO] En iyi skor (RF): {grid.best_score_}")

            self._plot_grid_search(grid, "Random Forest", embedding_name)
        except Exception as e:
            print(f"[ERROR] Random Forest eğitiminde hata: {e}")

    def train_svm(self, param_grid, embedding_name):
        """
        SVM modelini GridSearch ile eğitir.

        Args:
            param_grid (dict): Hiperparametre grid'i.
            embedding_name (str): Embedding modelinin adı.
        """
        print("\n[INFO] SVM GridSearchCV başlatılıyor...")
        svm = SVC(random_state=self.random_state)
        grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=0)

        try:
            grid.fit(self.X_train, self.y_train)
            self.models['SVM'] = grid.best_estimator_
            self.results['SVM'] = grid.best_score_

            print(f"[SUCCESS] En iyi parametreler (SVM): {grid.best_params_}")
            print(f"[INFO] En iyi skor (SVM): {grid.best_score_}")

            self._plot_grid_search(grid, "SVM", embedding_name)
        except Exception as e:
            print(f"[ERROR] SVM eğitiminde hata: {e}")

    def train_logreg(self, param_grid, embedding_name):
        """
        Logistic Regression modelini GridSearch ile eğitir.

        Args:
            param_grid (dict): Hiperparametre grid'i.
            embedding_name (str): Embedding modelinin adı.
        """
        print("\n[INFO] Logistic Regression GridSearchCV başlatılıyor...")
        logreg = LogisticRegression(max_iter=1000, random_state=self.random_state)
        grid = GridSearchCV(logreg, param_grid, cv=3, scoring='accuracy', verbose=0)

        try:
            grid.fit(self.X_train, self.y_train)
            self.models['Logistic Regression'] = grid.best_estimator_
            self.results['Logistic Regression'] = grid.best_score_

            print(f"[SUCCESS] En iyi parametreler (LogReg): {grid.best_params_}")
            print(f"[INFO] En iyi skor (LogReg): {grid.best_score_}")

            self._plot_grid_search(grid, "Logistic Regression", embedding_name)
        except Exception as e:
            print(f"[ERROR] Logistic Regression eğitiminde hata: {e}")

    def save_models(self, save_path, embedding_name):
        """
        Eğitilen modelleri dosyaya kaydeder.

        Args:
            save_path (str): Modellerin kaydedileceği yol.
            embedding_name (str): Embedding modelinin adı.
        """
        model_dir = f"{save_path}/{embedding_name.replace('/', '_')}"
        self._ensure_dir_exists(model_dir)

        for model_name, model in self.models.items():
            model_file = f"{model_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_file)
            print(f"[SUCCESS] Model kaydedildi: {model_file}")

    def visualize_results(self, embedding_name):
        """
        Eğitim sonuçlarını görselleştirir.

        Args:
            embedding_name (str): Embedding modelinin adı.
        """
        if not self.results:
            print("[INFO] Görselleştirilecek sonuç bulunamadı.")
            return

        plt.figure(figsize=(10, 6))
        bars = sns.barplot(x=list(self.results.keys()), y=list(self.results.values()), palette="viridis")
        plt.title(f"{embedding_name} - Model Karşılaştırması")
        plt.xlabel("Model")
        plt.ylabel("Doğruluk")
        plt.ylim(0, 1)

        for i, bar in enumerate(bars.patches):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        sanitized_embedding_name = embedding_name.replace("/", "_")
        plot_dir = f"train_results/{sanitized_embedding_name}"
        self._ensure_dir_exists(plot_dir)

        plot_path = f"{plot_dir}/accuracy_comparison.png"
        plt.savefig(plot_path)
        print(f"[SUCCESS] Accuracy Comparison görselleştirildi ve kaydedildi: {plot_path}")


def train_with_gridsearch(args):
    """
    Belirli bir embedding modeli ile GridSearch üzerinden eğitim yapar ve sonuçları kaydeder.

    Args:
        args (tuple): 
            - model_name (str): Embedding modelinin adı.
            - X_train (pd.Series): Eğitim veri metinleri.
            - y_train (pd.Series): Eğitim veri etiketleri.
            - X_test (pd.Series): Test veri metinleri.
            - y_test (pd.Series): Test veri etiketleri.
            - save_path (str): Modellerin kaydedileceği dizin.
            - gpu_device (torch.device): Kullanılacak cihaz (CPU veya GPU).
    """
    model_name, X_train, y_train, X_test, y_test, save_path, gpu_device = args
    print(f"[INFO] Başlatılıyor: {model_name} (GPU: {gpu_device})")

    # Embedding oluşturucu başlat
    embedder = EmbeddingGenerator(model_name, trust_remote_code=True, device=gpu_device)

    # Eğitim embedding'lerini oluştur
    print(f"[INFO] Embedding oluşturuluyor: {model_name}")
    X_train_embedded = embedder.get_representation(
        model_data=(embedder.tokenizer, embedder.model), texts=X_train.tolist()
    )
    X_test_embedded = embedder.get_representation(
        model_data=(embedder.tokenizer, embedder.model), texts=X_test.tolist()
    )

    # t-SNE Görselleştirme
    print(f"[INFO] t-SNE görselleştirmesi yapılıyor: {model_name}")
    embedder.visualize_tsne(X_train_embedded, y_train, title=f"t-SNE {model_name} (Train Set)")
    embedder.visualize_tsne(X_test_embedded, y_test, title=f"t-SNE {model_name} (Test Set)")

    # Model eğitim süreci
    trainer = Trainer(X_train_embedded, y_train, device=gpu_device)

    # Modeller ve hiperparametre grid'leri
    classifiers = [
        ("Random Forest", trainer.train_rf, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
        ("SVM", trainer.train_svm, {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1, 1]}),
        ("Logistic Regression", trainer.train_logreg, {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']})
    ]

    # Her model için eğitim
    for clf_name, train_method, params in classifiers:
        print(f"[INFO] Eğitim başlıyor: {clf_name} ({model_name})")
        train_method(params, embedding_name=model_name)

    # Modelleri kaydet
    trainer.save_models(save_path=save_path, embedding_name=model_name)

    # Sonuçları görselleştir
    trainer.visualize_results(embedding_name=model_name)

    print(f"[INFO] Tamamlandı: {model_name}")


if __name__ == "__main__":
    """
    Ana çalışma fonksiyonu. Paralel işlem kullanarak embedding'lerle model eğitimi yapar.
    """
    # Paralel işlemleri başlatmak için gerekli
    multiprocessing.set_start_method("spawn", force=True)

    # Veri işleme adımları
    dataset_name = "savasy/ttc4900"  # Veri seti ismi
    processor = DataProcessor(dataset_name, text_column='text', label_column='category')

    print("[INFO] Rastgele alt küme seçiliyor...")
    subset = processor.get_random_subset(subset_size=4900)

    print("[INFO] Veriyi eğitim ve test setine bölme...")
    X_train, X_test, y_train, y_test = processor.split_data(subset)

    # Kullanılacak embedding modelleri
    model_names = [
        # "jinaai/jina-embeddings-v3",
        "sentence-transformers/all-MiniLM-L12-v2",
        "intfloat/multilingual-e5-large-instruct",
        # "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v1",
        "dbmdz/bert-base-turkish-cased",
    ]
    save_path = "models"  # Modellerin kaydedileceği dizin
    os.makedirs(save_path, exist_ok=True)

    # Paralel işlem argümanlarını oluştur
    process_args = []
    for model_name in model_names:
        # CPU veya GPU cihazı ayarla
        if "jinaai" in model_name or "nomic" in model_name:
            device = torch.device("cpu")
            print(f"[INFO] {model_name} için CPU kullanılıyor.")
        else:
            device = torch_directml.device()
            print(f"[INFO] {model_name} için GPU kullanılıyor.")

        process_args.append((model_name, X_train, y_train, X_test, y_test, save_path, device))

    # Paralel işlem havuzu başlat ve GridSearch ile modelleri eğit
    print("[INFO] Paralel GridSearch başlatılıyor...")
    with multiprocessing.Pool(processes=len(model_names)) as pool:
        pool.map(train_with_gridsearch, process_args)

    print("[INFO] Tüm modeller başarıyla işlendi.")
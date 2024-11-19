import os
import joblib
import torch
from tqdm import tqdm
from dataset import DataProcessor
from embedding_generator import EmbeddingGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class Trainer:
    def __init__(self, X_train, y_train, device, random_state=42):
        """
        Trainer sınıfı, model eğitiminden sorumludur.
        Args:
            X_train: Eğitim veri özellikleri (embedding'ler)
            y_train: Eğitim etiketleri
            device: Kullanılacak cihaz (cuda veya cpu)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.device = device  # GPU/CPU cihazı
        self.random_state = random_state
        self.models = {}

    def train_rf(self, param_grid, embedding_name):
        """
        Random Forest modelini eğitir.
        """
        with tqdm(total=len(param_grid['n_estimators']), desc=f"Random Forest Eğitimi ({embedding_name})") as pbar:
            rf = RandomForestClassifier(random_state=self.random_state)
            grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
            grid.fit(self.X_train, self.y_train)
            self.models['Random Forest'] = grid.best_estimator_
            pbar.update(len(param_grid['n_estimators']))
        print(f"En iyi parametreler (RF): {grid.best_params_}")

    def train_svm(self, param_grid, embedding_name):
        """
        SVM modelini eğitir.
        """
        with tqdm(total=len(param_grid['C']), desc=f"SVM Eğitimi ({embedding_name})") as pbar:
            svm = SVC(random_state=self.random_state)
            grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
            grid.fit(self.X_train, self.y_train)
            self.models['SVM'] = grid.best_estimator_
            pbar.update(len(param_grid['C']))
        print(f"En iyi parametreler (SVM): {grid.best_params_}")

    def train_logreg(self, param_grid, embedding_name):
        """
        Logistic Regression modelini eğitir.
        """
        with tqdm(total=len(param_grid['C']), desc=f"Logistic Regression Eğitimi ({embedding_name})") as pbar:
            logreg = LogisticRegression(max_iter=1000, random_state=self.random_state)
            grid = GridSearchCV(logreg, param_grid, cv=3, scoring='accuracy')
            grid.fit(self.X_train, self.y_train)
            self.models['Logistic Regression'] = grid.best_estimator_
            pbar.update(len(param_grid['C']))
        print(f"En iyi parametreler (LogReg): {grid.best_params_}")

    def save_models(self, save_path, embedding_name):
        """
        Eğitilen modelleri dosyaya kaydeder.
        """
        for model_name, model in self.models.items():
            model_file = f"{save_path}/{embedding_name}_{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_file)
            print(f"Model kaydedildi: {model_file}")


if __name__ == "__main__":
    # GPU veya CPU cihazını seç
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılacak cihaz: {device}")

    # Veri seti adı
    dataset_name = "maydogan/TRSAv1"

    # Embedding modelleri
    model_names = [
        "jinaai/jina-embeddings-v3",
        "sentence-transformers/all-MiniLM-L12-v2",
        "intfloat/multilingual-e5-large-instruct",
        "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v1",
        "dbmdz/bert-base-turkish-cased",
    ]

    # Veri işlemleri
    print("Veri seti işleniyor...")
    processor = DataProcessor(dataset_name, text_column='review', label_column='score')
    subset = processor.get_random_subset(subset_size=5000)
    X_train, X_test, y_train, y_test = processor.split_data(subset)
    print("Veri işlemleri tamamlandı.")

    # Hiperparametre grid'leri
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1, 1]}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    logreg_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}

    # Model dosyalarını kaydetmek için klasör oluştur
    save_path = "models"
    os.makedirs(save_path, exist_ok=True)

    # Embedding modeli döngüsü
    for model_name in tqdm(model_names, desc="Embedding Modelleri"):
        print(f"\nEmbedding modeli: {model_name}")
        embedder = EmbeddingGenerator(model_name, trust_remote_code=True, device=device)

        # Eğitim seti embedding'lerini oluştur
        print(f"Embedding oluşturuluyor: {model_name}")
        X_train_embedded = embedder.encode(X_train.tolist(), pooling="mean")

        # Trainer
        trainer = Trainer(X_train_embedded, y_train, device=device)

        # Sınıflandırıcı döngüsü
        classifiers = [("Random Forest", trainer.train_rf, rf_params),
                       ("SVM", trainer.train_svm, svm_params),
                       ("Logistic Regression", trainer.train_logreg, logreg_params)]

        for clf_name, train_method, params in tqdm(classifiers, desc=f"Modeller ({model_name})"):
            print(f"{clf_name} eğitiliyor...")
            train_method(params, embedding_name=model_name)

        # Eğitilen modelleri kaydet
        trainer.save_models(save_path=save_path, embedding_name=model_name)

    print("\nTüm modeller kaydedildi!")

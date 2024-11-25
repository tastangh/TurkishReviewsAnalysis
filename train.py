import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset import DataProcessor
from embedding_generator import EmbeddingGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
import torch_directml 
import multiprocessing


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
        self.results = {}  # Eğitim sonuçlarını sakla

    def _ensure_dir_exists(self, path):
        """Klasör var mı kontrol eder, yoksa oluşturur."""
        os.makedirs(path, exist_ok=True)

    def _plot_grid_search(self, grid, model_name, embedding_name):
        """GridSearchCV sonuçlarını görselleştirir."""
        results = grid.cv_results_
        mean_scores = results['mean_test_score']
        params = results['params']

        # Performansları çubuğun uzunluğuna göre sıralayarak çiz
        sorted_indices = sorted(range(len(mean_scores)), key=lambda i: mean_scores[i], reverse=True)
        sorted_scores = [mean_scores[i] for i in sorted_indices]
        sorted_params = [params[i] for i in sorted_indices]

        plt.figure(figsize=(12, 6))
        barplot = sns.barplot(x=sorted_scores, y=[str(p) for p in sorted_params], palette="viridis")
        plt.title(f"{model_name} (Embedding: {embedding_name}) - Grid Search Results")
        plt.xlabel("Mean Test Accuracy")
        plt.ylabel("Hyperparameters")
        plt.xlim(0, 1)  # x ekseni 0 ile 1 arasında olmalı

        # Add text annotations on top of the bars
        for i, bar in enumerate(barplot.patches):
            x = bar.get_width()  # The length of the bar (accuracy value)
            y = bar.get_y() + bar.get_height() / 2  # Centered vertically
            plt.text(
                x + 0.01,  # Slightly to the right of the bar edge
                y,         # Centered vertically
                f"%{x * 100:.3f}",  # Format score as percentage
                ha='left', va='center', fontsize=10, color='black'
            )

        plt.tight_layout()

        # Save and show
        sanitized_embedding_name = embedding_name.replace("/", "_")  # Replace forbidden characters in filenames
        sanitized_model_name = model_name.lower().replace(" ", "_")
        plot_dir = f"plots/{sanitized_embedding_name}"
        self._ensure_dir_exists(plot_dir)

        plot_path = f"{plot_dir}/{sanitized_model_name}_grid_search.png"
        plt.savefig(plot_path)
        print(f"[SUCCESS] Grid Search görselleştirildi ve kaydedildi: {plot_path}")
        # plt.show()


    def train_rf(self, param_grid, embedding_name):
        """
        Random Forest modelini eğitir.
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

            # Grid Search görselleştir
            self._plot_grid_search(grid, "Random Forest", embedding_name)
        except Exception as e:
            print(f"[ERROR] Random Forest eğitiminde hata: {e}")

    def train_svm(self, param_grid, embedding_name):
        """
        SVM modelini eğitir.
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

            # Grid Search görselleştir
            self._plot_grid_search(grid, "SVM", embedding_name)
        except Exception as e:
            print(f"[ERROR] SVM eğitiminde hata: {e}")

    def train_logreg(self, param_grid, embedding_name):
        """
        Logistic Regression modelini eğitir.
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

            # Grid Search görselleştir
            self._plot_grid_search(grid, "Logistic Regression", embedding_name)
        except Exception as e:
            print(f"[ERROR] Logistic Regression eğitiminde hata: {e}")

    def save_models(self, save_path, embedding_name):
        """
        Eğitilen modelleri dosyaya kaydeder.
        """
        model_dir = f"{save_path}/{embedding_name.replace('/', '_')}"
        self._ensure_dir_exists(model_dir)

        for model_name, model in self.models.items():
            model_file = f"{model_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_file)
            print(f"[SUCCESS] Model kaydedildi: {model_file}")

    def visualize_results(self, embedding_name):
            """Eğitim sonuçlarını görselleştirir."""
            if not self.results:
                print("[INFO] Görselleştirilecek sonuç bulunamadı.")
                return

            plt.figure(figsize=(10, 6))
            bars = sns.barplot(x=list(self.results.keys()), y=list(self.results.values()), palette="viridis")
            plt.title(f"{embedding_name} - Model Karşılaştırması")
            plt.xlabel("Model")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)  # Y eksenini 0-1 arasında ayarla

            # Add accuracy values on top of each bar
            for i, bar in enumerate(bars.patches):
                x = bar.get_x() + bar.get_width() / 2  # Center of the bar
                y = bar.get_height()  # Height of the bar (accuracy value)
                plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=10)

            plt.tight_layout()

            # Kaydet ve göster
            sanitized_embedding_name = embedding_name.replace("/", "_")
            plot_dir = f"plots/{sanitized_embedding_name}"
            self._ensure_dir_exists(plot_dir)

            plot_path = f"{plot_dir}/accuracy_comparison.png"
            plt.savefig(plot_path)
            print(f"[SUCCESS] Accuracy Comparison görselleştirildi ve kaydedildi: {plot_path}")
            # plt.show()

# Define a function to handle embedding and training for a specific model
def process_embedding_model(model_name, X_train, y_train, X_test, y_test, save_path, device):
    print(f"\n[INFO] Processing embedding model: {model_name}")
    embedder = EmbeddingGenerator(model_name, trust_remote_code=True, device=device)

    # Create embeddings
    print(f"[INFO] Generating embeddings for model: {model_name}")
    X_train_embedded = embedder.get_representation(
        model_data=(embedder.tokenizer, embedder.model), texts=X_train.tolist()
    )
    X_test_embedded = embedder.get_representation(
        model_data=(embedder.tokenizer, embedder.model), texts=X_test.tolist()
    )

    # t-SNE Visualization
    print(f"[INFO] Generating t-SNE visualization for: {model_name}")
    embedder.visualize_tsne(X_train_embedded, y_train, title=f"t-SNE {model_name} (Train Set)")
    embedder.visualize_tsne(X_test_embedded, y_test, title=f"t-SNE {model_name} (Test Set)")

    # Train models
    trainer = Trainer(X_train_embedded, y_train, device=device)
    classifiers = [
        ("Random Forest", trainer.train_rf, {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
        ("SVM", trainer.train_svm, {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1, 1]}),
        ("Logistic Regression", trainer.train_logreg, {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']})
    ]

    for clf_name, train_method, params in classifiers:
        print(f"[INFO] Training {clf_name} with model: {model_name}")
        train_method(params, embedding_name=model_name)

    # Save trained models
    trainer.save_models(save_path=save_path, embedding_name=model_name)

    # Visualize results
    trainer.visualize_results(embedding_name=model_name)

    print(f"[INFO] Completed processing for model: {model_name}")

if __name__ == "__main__":
    # Select device
    device = torch_directml.device()
    print(f"[INFO] Using device: {device}")

    # Dataset name
    dataset_name = "maydogan/TRSAv1"

    # Data processing
    print("[INFO] Processing dataset...")
    processor = DataProcessor(dataset_name, text_column='review', label_column='score')

    # Random subset selection
    print("[INFO] Selecting random subset...")
    subset = processor.get_random_subset(subset_size=5000)
    print("[INFO] Subset class distribution:")
    print(subset['score'].value_counts())

    # Data splitting
    print("[INFO] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = processor.split_data(subset)

    # Model names and save path
    model_names = [
        # "jinaai/jina-embeddings-v3",
        # "sentence-transformers/all-MiniLM-L12-v2",
        # "intfloat/multilingual-e5-large-instruct",
        # "BAAI/bge-m3",
        # "nomic-ai/nomic-embed-text-v1",
        "dbmdz/bert-base-turkish-cased",
    ]
    save_path = "models"
    os.makedirs(save_path, exist_ok=True)

    # Create a process for each model
    processes = []
    for model_name in model_names:
        process = multiprocessing.Process(
            target=process_embedding_model,
            args=(model_name, X_train, y_train, X_test, y_test, save_path, device)
        )
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("\n[INFO] All models processed and saved successfully!")
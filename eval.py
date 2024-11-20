import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Eval:
    def __init__(self, model_dir, X_test, y_test):
        """
        Model değerlendirme sınıfı.
        
        Args:
            model_dir: Modellerin kaydedildiği dizin.
            X_test: Test veri özellikleri.
            y_test: Test etiketleri.
        """
        self.model_dir = model_dir
        self.X_test = X_test
        self.y_test = y_test
        self.models = self._load_models()

    def _load_models(self):
        """
        Kaydedilen modelleri yükler.

        Returns:
            dict: Model adlarını ve modelleri içeren bir sözlük.
        """
        models = {}
        for file in os.listdir(self.model_dir):
            if file.endswith(".pkl"):
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(self.model_dir, file)
                models[model_name] = joblib.load(model_path)
                print(f"[INFO] Model yüklendi: {model_name}")
        return models

    def _plot_confusion_matrix(self, cm, model_name):
        """Confusion Matrix'i görselleştirir."""
        # Benzersiz sınıf isimlerini y_test'ten al
        class_names = np.unique(self.y_test)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()

        # Kaydet ve göster
        plot_path = f"plots/confusion_matrix_{model_name}.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_path)
        print(f"[SUCCESS] Confusion Matrix kaydedildi: {plot_path}")
        plt.show()

    def evaluate_models(self):
        """
        Yüklenen modelleri değerlendirir ve metrikleri döndürür.
        """
        results = {}
        for model_name, model in self.models.items():
            print(f"\n[INFO] Model değerlendiriliyor: {model_name}")
            y_pred = model.predict(self.X_test)

            # Accuracy
            acc = accuracy_score(self.y_test, y_pred)

            # Classification Report
            report = classification_report(self.y_test, y_pred)

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            self._plot_confusion_matrix(cm, model_name)

            print(f"[RESULTS] Accuracy: {acc}")
            print(f"[RESULTS] Classification Report:\n{report}")

            results[model_name] = {
                "accuracy": acc,
                "classification_report": report,
                "confusion_matrix": cm
            }
        return results

    def visualize_accuracies(self, results):
        """
        Modellerin doğruluk oranlarını bar grafikte gösterir.
        """
        model_names = list(results.keys())
        accuracies = [metrics["accuracy"] for metrics in results.values()]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=accuracies, palette="viridis")
        plt.title("Model Accuracy Comparison")
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)

        # Çubukların üzerine yüzdelik skor ekle
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Kaydet ve göster
        plot_path = "plots/accuracy_comparison.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(plot_path)
        print(f"[SUCCESS] Accuracy Comparison grafiği kaydedildi: {plot_path}")
        plt.show()


if __name__ == "__main__":
    import torch
    from dataset import DataProcessor
    from embedding_generator import EmbeddingGenerator

    # GPU veya CPU cihazını seç
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Kullanılacak cihaz: {device}")

    # Veri seti adı
    dataset_name = "maydogan/TRSAv1"

    # Embedding modeli
    model_name = "dbmdz/bert-base-turkish-cased"

    # Veri işlemleri
    print("[INFO] Veri seti işleniyor...")
    processor = DataProcessor(dataset_name, text_column='review', label_column='score')
    subset = processor.get_random_subset(subset_size=5000)
    X_train, X_test, y_train, y_test = processor.split_data(subset)
    print("[INFO] Veri işlemleri tamamlandı.")

    # Embedding oluştur
    embedder = EmbeddingGenerator(model_name, trust_remote_code=True, device=device)
    X_test_embedded = embedder.encode(X_test.tolist(), pooling="mean")

    # Modellerin kaydedildiği yol
    model_dir = "models/dbmdz/bert-base-turkish-cased"

    # Eval sınıfını kullanarak modelleri değerlendir
    evaluator = Eval(model_dir, X_test_embedded, y_test)
    evaluation_results = evaluator.evaluate_models()

    # Accuracy görselleştirme
    evaluator.visualize_accuracies(evaluation_results)

    # Sonuçları kaydet veya yazdır
    for model_name, metrics in evaluation_results.items():
        print(f"\n[SUMMARY] Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Classification Report:\n{metrics['classification_report']}")

# TODO : üretilen png'ler model bazlı kaydedilmeli
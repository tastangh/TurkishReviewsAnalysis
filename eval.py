import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from embedding_generator import EmbeddingGenerator


class Eval:
    def __init__(self, model_dirs, embedding_model_names, trust_remote_code, X_test_texts, y_test, device):
        """
        Birden fazla modeli değerlendirme sınıfı.
        
        Args:
            model_dirs: Modellerin kaydedildiği dizinlerin listesi.
            embedding_model_names: Kullanılacak embedding model adlarının listesi.
            trust_remote_code: Embedding modelini indirirken güven seçeneği.
            X_test_texts: Test veri metinleri.
            y_test: Test etiketleri.
            device: Kullanılacak cihaz (cuda/cpu).
        """
        self.model_dirs = model_dirs
        self.embedding_model_names = embedding_model_names
        self.X_test_texts = X_test_texts
        self.y_test = y_test
        self.device = device
        self.models = self._load_models()

    def _load_models(self):
        """
        Kaydedilen modelleri yükler.

        Returns:
            dict: Model adlarını ve modelleri içeren bir sözlük.
        """
        models = {}
        for model_dir in self.model_dirs:
            for file in os.listdir(model_dir):
                if file.endswith(".pkl"):
                    model_name = os.path.basename(model_dir)
                    training_method = os.path.splitext(file)[0]  # Eğitim metodu: svm, random_forest, vs.
                    full_model_name = f"{model_name}_{training_method}"
                    model_path = os.path.join(model_dir, file)
                    models[full_model_name] = joblib.load(model_path)
                    print(f"[INFO] Model yüklendi: {full_model_name}")
        return models

    def _ensure_eval_dir(self, model_name):
        """
        Belirtilen model için evals altında bir dizin oluşturur.
        
        Args:
            model_name: Model adı (klasör adı olarak kullanılacak).
        Returns:
            str: Model için oluşturulan dizin yolu.
        """
        model_eval_dir = os.path.join("evals", model_name)
        os.makedirs(model_eval_dir, exist_ok=True)
        return model_eval_dir

    def _plot_confusion_matrix(self, cm, model_name, training_method, output_dir):
        """Confusion Matrix'i görselleştirir ve kaydeder."""
        class_names = np.unique(self.y_test)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.title(f"Confusion Matrix - {training_method}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"confusion_matrix_{training_method}.png")
        plt.savefig(plot_path)
        print(f"[SUCCESS] Confusion Matrix kaydedildi: {plot_path}")
        plt.close()

    def _visualize_accuracy(self, model_results, model_name, output_dir):
        """
        Modelin doğruluk oranlarını bar grafikte gösterir ve kaydeder.
        
        Args:
            model_results: Modelin farklı eğitim metotları için sonuçları.
            model_name: Modelin adı.
            output_dir: Kaydedilecek dizin.
        """
        training_methods = list(model_results.keys())
        accuracies = [metrics["accuracy"] for metrics in model_results.values()]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=training_methods, y=accuracies, palette="viridis")
        plt.title(f"Doğruluk Karşılaştırması - {model_name}")
        plt.xlabel("Eğitim Türü")
        plt.ylabel("Başarı")
        plt.ylim(0, 1)

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', va='bottom', fontsize=10)

        plot_path = os.path.join(output_dir, "accuracy_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"[SUCCESS] Doğruluk Karşılaştırması kaydedildi: {plot_path}")
        plt.close()

    def _save_classification_report(self, report, model_name, training_method, output_dir):
        """
        Classification report'u bir metin dosyasına kaydeder.
        
        Args:
            report: Classification report (string formatında).
            model_name: Modelin adı.
            training_method: Eğitim yöntemi (örneğin, svm, random_forest).
            output_dir: Kaydedilecek dizin.
        """
        file_path = os.path.join(output_dir, f"classification_report_{training_method}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Eğitim Türü: {training_method}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
        print(f"[SUCCESS] Classification Report kaydedildi: {file_path}")

    def evaluate_models(self):
        """
        Yüklenen modelleri değerlendirir ve metrikleri döndürür.
        """
        results = {}

        for model_dir, embedding_model_name in zip(self.model_dirs, self.embedding_model_names):
            print(f"[INFO] Test embedding oluşturuluyor: {embedding_model_name}")
            embedder = EmbeddingGenerator(embedding_model_name, trust_remote_code=True, device=self.device)
            X_test_embedded = embedder.get_representation(
                model_data=(embedder.tokenizer, embedder.model), texts=self.X_test_texts
            )

            # Model adı ve sonuçları saklanacak
            model_name = os.path.basename(model_dir)
            model_eval_dir = self._ensure_eval_dir(model_name)
            model_results = {}

            for full_model_name, model in self.models.items():
                if full_model_name.startswith(model_name):  # İlgili modele ait modeller
                    training_method = full_model_name.split("_")[-1]
                    print(f"\n[INFO] Model değerlendiriliyor: {training_method}")

                    # Tahmin ve metrikler
                    y_pred = model.predict(X_test_embedded)
                    acc = accuracy_score(self.y_test, y_pred)
                    report = classification_report(self.y_test, y_pred, zero_division=0)
                    cm = confusion_matrix(self.y_test, y_pred)

                    # Confusion Matrix görsellerini kaydet
                    self._plot_confusion_matrix(cm, model_name, training_method, model_eval_dir)

                    # Classification Report'u kaydet
                    self._save_classification_report(report, model_name, training_method, model_eval_dir)

                    print(f"[RESULTS] {training_method} Accuracy: {acc}")
                    print(f"[RESULTS] {training_method} Classification Report:\n{report}")

                    model_results[training_method] = {
                        "accuracy": acc,
                        "classification_report": report,
                        "confusion_matrix": cm
                    }

            # Accuracy Comparison görselini kaydet
            self._visualize_accuracy(model_results, model_name, model_eval_dir)
            results[model_name] = model_results

        return results


if __name__ == "__main__":
    import torch
    from dataset import DataProcessor

    device = torch.device("cpu")
    print(f"[INFO] Kullanılacak cihaz: {device}")

    dataset_name = "maydogan/TRSAv1"
    embedding_model_names = [
        "jinaai/jina-embeddings-v3",
        "sentence-transformers/all-MiniLM-L12-v2",
        "intfloat/multilingual-e5-large-instruct",
        "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v1",
        "dbmdz/bert-base-turkish-cased",
    ]
    model_dirs = [
        "models/jinaai_jina-embeddings-v3",
        "models/sentence-transformers_all-MiniLM-L12-v2",
        "models/intfloat_multilingual-e5-large-instruct",
        "models/BAAI_bge-m3",
        "models/nomic-ai_nomic-embed-text-v1",
        "models/dbmdz_bert-base-turkish-cased",
    ]

    print("[INFO] Veri seti işleniyor...")
    processor = DataProcessor(dataset_name, text_column='review', label_column='score')
    subset = processor.get_random_subset(subset_size=5000)
    X_train, X_test, y_train, y_test = processor.split_data(subset)
    print("[INFO] Veri işlemleri tamamlandı.")

    evaluator = Eval(
        model_dirs=model_dirs,
        embedding_model_names=embedding_model_names,
        trust_remote_code=True,
        X_test_texts=X_test.tolist(),
        y_test=y_test,
        device=device
    )

    evaluation_results = evaluator.evaluate_models()

    for model_name, training_methods in evaluation_results.items():
        print(f"\n[SUMMARY] Model: {model_name}")
        for training_method, metrics in training_methods.items():
            print(f"  Training Method: {training_method}")
            print(f"  Accuracy: {metrics['accuracy']}")
            print(f"  Classification Report:\n{metrics['classification_report']}")

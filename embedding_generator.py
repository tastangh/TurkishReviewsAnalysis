import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns


class EmbeddingGenerator:
    """
    Metinlerden gömülü vektörler üretmek ve bu vektörleri t-SNE ile görselleştirmek için bir sınıf.

    Attributes:
        model_name (str): Kullanılacak dil modelinin adı (ör. "bert-base-uncased").
        device (str): Modelin çalıştırılacağı cihaz (ör. "cuda" veya "cpu").
        tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer objesi.
        model (transformers.PreTrainedModel): Hugging Face dil modeli.
    """

    def __init__(self, model_name, trust_remote_code=False, device=None):
        """
        EmbeddingGenerator sınıfının yapıcı metodu.

        Args:
            model_name (str): Yüklenmek istenen dil modelinin adı.
            trust_remote_code (bool): Uzak kodların çalıştırılmasına izin verilmesi (varsayılan: False).
            device (str): Modelin çalıştırılacağı cihaz (ör. "cuda" veya "cpu").
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(self.device)

    def get_representation(self, model_data, texts):
        """
        Verilen metinlerden gömülü vektörler üretir.

        Args:
            model_data (tuple): Tokenizer ve model objelerinden oluşan bir tuple.
            texts (list): Gömülü vektörleri üretilecek metinlerin listesi.

        Returns:
            np.ndarray: Metinlerden üretilen gömülü vektörlerin numpy dizisi.
        """
        tokenizer, model = model_data
        representations = []
        for text in tqdm(texts, desc="Metinler işleniyor", leave=False):
            # Metinleri tokenize et ve modele uygun hale getir
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state.mean(dim=1).float()  # Ortalama alınmış son katman
            representations.append(last_hidden_state.cpu().numpy())
        return np.vstack(representations)

    def visualize_tsne(self, embeddings, labels, title="t-SNE Görselleştirmesi", save_dir="plots"):
        """
        Gömülü vektörleri t-SNE algoritmasıyla görselleştirir.

        Args:
            embeddings (np.ndarray): Gömülü vektörlerin numpy dizisi (şekil: [n_samples, n_features]).
            labels (list veya np.ndarray): Gömülü vektörlere karşılık gelen etiketler.
            title (str): Grafik başlığı.
            save_dir (str): Grafiğin kaydedileceği dizin.
        """
        # t-SNE ile boyut indirgeme
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Görselleştirme işlemi
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
            hue=labels, palette="viridis", s=50, legend="full"
        )
        plt.title(title)
        plt.xlabel("t-SNE Boyut 1")
        plt.ylabel("t-SNE Boyut 2")

        # Dosya adı için başlığı düzenle
        sanitized_title = title.replace(" ", "_").replace("/", "_").lower()
        sanitized_dir = os.path.join(save_dir, sanitized_title.split("_")[0])  # t-SNE alt klasörü oluştur
        os.makedirs(sanitized_dir, exist_ok=True)

        filename = os.path.join(sanitized_dir, f"{sanitized_title}.png")
        plt.savefig(filename, dpi=300)
        print(f"[INFO] t-SNE grafiği kaydedildi: {filename}")

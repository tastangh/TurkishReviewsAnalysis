import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class EmbeddingGenerator:
    def __init__(self, model_name, trust_remote_code=False, device=None):
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(self.device)

    def encode(self, texts, pooling="mean"):
        embeddings = []
        for text in tqdm(texts, desc="Embedding İşlemi"):
            try:
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                if pooling == "mean":
                    embedding = outputs.last_hidden_state.mean(dim=1)
                elif pooling == "max":
                    embedding, _ = outputs.last_hidden_state.max(dim=1)
                elif pooling == "cls":
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Havuzlama yöntemi 'mean', 'max' veya 'cls' olmalı.")

                embeddings.append(embedding.cpu().float().numpy())
            except Exception as e:
                print(f"Hata: '{text}' metni işlenirken sorun oluştu -> {e}")
        return np.vstack(embeddings)

    def visualize_embeddings(self, embeddings, labels, method="pca"):
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2)
        else:
            raise ValueError("Yalnızca 'pca' veya 'tsne' destekleniyor.")

        reduced = reducer.fit_transform(embeddings)
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', s=5)
        plt.title(f"{method.upper()} ile Görselleştirme")
        plt.colorbar()
        plt.show()

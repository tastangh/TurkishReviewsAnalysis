import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns 

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

    def visualize_tsne(self, embeddings, labels, title="t-SNE Visualization", save_dir="plots"):
        """
        Performs t-SNE visualization of the embeddings.
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            labels: list or numpy array of labels corresponding to embeddings
            title: Title of the plot
            save_dir: Directory where the plot will be saved
        """
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
            hue=labels, palette="viridis", s=50, legend="full"
        )
        plt.title(title)
        plt.xlabel("t-SNE Boyut 1")
        plt.ylabel("t-SNE Boyut 2")

        # Sanitize title to create a valid filename
        sanitized_title = title.replace(" ", "_").replace("/", "_").lower()
        sanitized_dir = os.path.join(save_dir, sanitized_title.split("_")[0])  # Create a subfolder for t-SNE plots
        os.makedirs(sanitized_dir, exist_ok=True)

        filename = os.path.join(sanitized_dir, f"{sanitized_title}.png")
        plt.savefig(filename, dpi=300)
        print(f"[INFO] t-SNE plot saved at {filename}")
        # plt.show()

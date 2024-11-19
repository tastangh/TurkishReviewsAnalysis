from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name):
        """
        EmbeddingGenerator, verilen model adıyla metinlerden embedding çıkarır.
        
        Args:
            model_name (str): Hugging Face model ismi (örneğin, 'dbmdz/bert-base-turkish-cased')
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts, pooling="mean"):
        """
        Metinleri embedding vektörlerine dönüştürür.
        
        Args:
            texts (list of str): İşlenecek metinlerin listesi
            pooling (str): 'mean', 'max', veya 'cls' seçeneğiyle embedding havuzlama yöntemi
        
        Returns:
            np.ndarray: Metinlerin embedding vektörleri
        """
        # Tokenize metinler
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Modeli çalıştır
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Transformer çıktılarını al (hidden states)
        embeddings = outputs.last_hidden_state

        # Havuzlama (pooling) işlemi
        if pooling == "mean":
            embeddings = embeddings.mean(dim=1)
        elif pooling == "max":
            embeddings, _ = embeddings.max(dim=1)
        elif pooling == "cls":
            embeddings = embeddings[:, 0, :]  # CLS token'ını al
        else:
            raise ValueError("Havuzlama yöntemi 'mean', 'max' veya 'cls' olmalı.")
        
        return embeddings.cpu().numpy() 
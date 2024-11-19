import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import traceback

class EmbeddingGenerator:
    def __init__(self, model_name, trust_remote_code=False, device=None):
        """
        EmbeddingGenerator, verilen model adıyla metinlerden embedding çıkarır.
        
        Args:
            model_name (str): Hugging Face model ismi
            trust_remote_code (bool): Hugging Face modeli için özel kod yüklemesine izin ver (default: False)
            device (torch.device): Kullanılacak cihaz (cuda veya cpu)
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model {self.device} cihazında yüklenecek.")

        # Tokenizer ve model yükleme
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(self.device)

    def encode(self, texts, pooling="mean"):
        """
        Metinleri seçilen modelle embedding'e dönüştürür.

        Args:
            texts (list of str): Metin listesi
            pooling (str): 'mean', 'max' veya 'cls' pooling yöntemi

        Returns:
            np.ndarray: Metinlerin embedding vektörleri
        """
        embeddings = []

        for text in tqdm(texts, desc="Embedding İşlemi"):
            try:
                # Tokenizer'ı kullanarak metni tensöre dönüştür
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)

                # Modeli çalıştır
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Havuzlama (pooling)
                if pooling == "mean":
                    embedding = outputs.last_hidden_state.mean(dim=1)
                elif pooling == "max":
                    embedding, _ = outputs.last_hidden_state.max(dim=1)
                elif pooling == "cls":
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Havuzlama yöntemi 'mean', 'max' veya 'cls' olmalı.")

                # Embedding'i float32 olarak CPU'ya taşı ve numpy formatına çevir
                embeddings.append(embedding.cpu().float().numpy())

            except Exception as e:
                print(f"Hata: '{text}' metni işlenirken sorun oluştu -> {e}")
                traceback.print_exc()

        return np.vstack(embeddings)

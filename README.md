

## Kurulum

### 1. Sanal Ortam Oluşturma
Öncelikle bir sanal ortam oluşturun ve aktive edin:
```bash
python -m venv reviews_env
reviews_env\Scripts\activate  # Windows
source reviews_env/bin/activate  # MacOS/Linux
```

### 2. Gereksinimleri Yükleme
Projeye ait bağımlılıkları yüklemek için:
```bash
pip install -r requirements.txt
```

Eğer DirectML GPU kullanımı için ek bir bağımlılığa ihtiyacınız varsa, aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements_directml.txt
```

---

## Kullanım

### 1. Eğitim
Eğitim aşamasını başlatmak için aşağıdaki komutu çalıştırın. Bu komut, çeşitli embedding modelleri ile temsil edilen veriler üzerinden RF, SVM ve Logistic Regression modellerini eğitir ve sonuçları kaydeder:
```bash
python train.py
```

- Eğitim sırasında, modellerin en iyi hiperparametrelerini bulmak için `GridSearchCV` kullanılır.
- Eğitilen modeller `models` klasörü altında kaydedilir.

### 2. Değerlendirme
Test kümesi üzerinden modellerin performansını değerlendirmek için:
```bash
python eval.py
```

Değerlendirme sonuçları `evals` klasöründe saklanır:
- **Doğruluk Karşılaştırması (Accuracy Comparison):** Her model için ayrı bir doğruluk grafiği (`accuracy_comparison.png`).
- **Karmaşıklık Matrisi (Confusion Matrix):** Her modelin tahmin performansı (`confusion_matrix_<training_method>.png`).
- **Classification Report:** Test sonuçlarının ayrıntılı bir özeti (`classification_report_<training_method>.txt`).

---

## Çıktılar
Aşağıdaki klasörlerde proje çıktıları saklanır:

- `models/`: Eğitilen modellerin kayıtlı olduğu klasör.
- `evals/`: Değerlendirme sonuçlarının (grafikler, raporlar) saklandığı klasör.
- `plots/`: Embedding modellerinin görselleştirilmesi (ör. t-SNE).

2. **Eğitim:**
   ```bash
   python train.py
   ```

3. **Değerlendirme:**
   ```bash
   python eval.py
   ```

---

## Bağımlılıklar
Aşağıdaki bağımlılıkları yüklemek için `requirements.txt` dosyasını kullanabilirsiniz:

```plaintext
torch==2.0.1
transformers==4.30.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
numpy==1.24.3
tqdm==4.65.0
datasets==2.12.0
joblib==1.3.2
```
!pip install --upgrade transformers huggingface_hub

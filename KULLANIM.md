# 🚀 Projeyi Çalıştırma Rehberi

Bu dosya, projeyi adım adım nasıl çalıştıracağınızı gösterir.

## 📋 Hızlı Başlangıç (Windows)

### Yöntem 1: Otomatik Script (Önerilen)

1. **Çift tıklayın**: `run_project.bat` dosyasına çift tıklayın
2. **Bekleyin**: Script otomatik olarak:
   - Conda environment oluşturacak
   - Gereksinimleri yükleyecek
   - Modeli eğitecek

### Yöntem 2: Manuel Adımlar

#### Adım 1: Conda Environment Oluştur

PowerShell veya Command Prompt'u açın ve proje klasörüne gidin:

```bash
cd "C:\sinir aglari"
```

Conda environment oluşturun:

```bash
conda env create -f environment.yml
```

#### Adım 2: Environment'ı Aktifleştir

```bash
conda activate brain_tumor_cnn
```

#### Adım 3: Modeli Eğit

```bash
python train.py
```

## 📊 Çıktıları Nerede Göreceksiniz?

### 1. Terminal/Console Çıktısı

Eğitim sırasında terminalde şunları göreceksiniz:

```
Using device: cuda (veya cpu)
Loading dataset...
Total images found: 500+
  - Class 'yes' (tumor): 250
  - Class 'no' (no tumor): 250

Dataset split:
  - Training: 350 images
  - Validation: 50 images
  - Test: 100 images

Starting training for 30 epochs...
============================================================

Epoch 1/30
------------------------------------------------------------
Training: 100%|██████████| 11/11 [00:15<00:00, loss=0.6234, acc=65.23%]
Validation: 100%|██████████| 2/2 [00:02<00:00, loss=0.5123, acc=72.50%]

Epoch 1 Summary:
  Train Loss: 0.6234 | Train Acc: 65.23%
  Val Loss: 0.5123 | Val Acc: 72.50%
  Learning Rate: 0.001000
  ✓ Best model saved! (Val Acc: 72.50%)

... (30 epoch boyunca devam eder)

============================================================
Training completed in 12.34 minutes
Best validation accuracy: 89.50%

Loading best model for testing...

============================================================
TEST RESULTS
============================================================
Test Loss: 0.3456
Test Accuracy: 87.50%

Classification Report:
              precision    recall  f1-score   support

    No Tumor       0.89      0.85      0.87        50
       Tumor       0.86      0.90      0.88        50

    accuracy                           0.88       100
   macro avg       0.88      0.88      0.88       100
weighted avg       0.88      0.88      0.88       100

Confusion Matrix:
                Predicted
              No Tumor  Tumor
Actual No Tumor     43      7
Actual Tumor          6     44

Training history saved to training_history.png
```

### 2. Oluşturulan Dosyalar

Eğitim tamamlandıktan sonra şu dosyalar oluşacak:

#### 📁 `models/` klasörü
- **`brain_tumor_cnn.pth`**: Eğitilmiş model dosyası
  - Bu dosyayı `predict.py` ile kullanabilirsiniz

#### 📊 `training_history.png`
- Eğitim ve validation loss grafikleri
- Eğitim ve validation accuracy grafikleri
- Bu dosyayı herhangi bir görüntüleyici ile açabilirsiniz

### 3. Model Dosyası Detayları

`models/brain_tumor_cnn.pth` dosyası şunları içerir:
- Model ağırlıkları (weights)
- Optimizer durumu
- En iyi validation accuracy
- Eğitim konfigürasyonu

## 🔮 Tahmin Yapma (Model Eğitildikten Sonra)

Model eğitildikten sonra, yeni görüntüler üzerinde tahmin yapabilirsiniz:

### Tek Görüntü Tahmini

```bash
# Environment aktif olmalı
conda activate brain_tumor_cnn

# Tahmin yap
python predict.py --image dataset/yes/Y1.jpg
```

**Çıktı örneği:**

```
Using device: cuda
Loading model from models/brain_tumor_cnn.pth...
Model loaded successfully!

Processing image: dataset/yes/Y1.jpg
Making prediction...

============================================================
PREDICTION RESULTS
============================================================
Predicted Class: Tumor
Confidence: 94.23%

Class Probabilities:
  No Tumor: 5.77%
  Tumor:    94.23%
============================================================
```

### Farklı Görüntüler Test Etme

```bash
# Tümör olan görüntü
python predict.py --image dataset/yes/Y10.jpg

# Tümör olmayan görüntü
python predict.py --image dataset/no/1\ no.jpeg
```

## ⚠️ Sorun Giderme

### Problem 1: "conda: command not found"

**Çözüm**: Anaconda veya Miniconda kurulu değil. 
- [Anaconda İndir](https://www.anaconda.com/download)
- Kurulumdan sonra terminali yeniden başlatın

### Problem 2: "CUDA out of memory"

**Çözüm**: Batch size'ı küçültün. `train.py` dosyasında:
```python
config = {
    'batch_size': 16,  # 32 yerine 16 yapın
    ...
}
```

### Problem 3: "ModuleNotFoundError"

**Çözüm**: Environment aktif değil veya paketler kurulmamış:
```bash
conda activate brain_tumor_cnn
pip install -r requirements.txt
```

### Problem 4: Veri seti bulunamıyor

**Çözüm**: `dataset/` klasörünün proje kök dizininde olduğundan emin olun.

## 📈 Eğitim Süresi

- **CPU**: ~2-4 saat (30 epoch)
- **GPU (NVIDIA)**: ~10-30 dakika (30 epoch)

## 🎯 Sonraki Adımlar

1. ✅ Modeli eğitin (`python train.py`)
2. ✅ `training_history.png` dosyasını kontrol edin
3. ✅ Test sonuçlarını inceleyin
4. ✅ Yeni görüntülerle tahmin yapın (`python predict.py`)

## 💡 İpuçları

- İlk eğitimde epoch sayısını azaltabilirsiniz (test için):
  ```python
  'num_epochs': 5,  # Hızlı test için
  ```
- GPU kullanımı için CUDA kurulu olmalı
- Eğitim sırasında `Ctrl+C` ile durdurabilirsiniz (model kaydedilir)

---

**Sorularınız için**: README.md dosyasına bakın veya issue açın.







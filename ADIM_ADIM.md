# Projeyi Çalıştırma - Adım Adım Rehber

### Yöntem 1: Otomatik Script

1. **`BASLAT.bat`** dosyasına çift tıklayın
2. Bekleyin - Model otomatik olarak eğitilecek
3. Bittiğinde `models/brain_tumor_cnn.pth` dosyası oluşacak

---

## Manuel Yöntem (Terminal)

### Adım 1: Terminal Açın

**Windows:**
- `Windows + R` tuşlarına basın
- `powershell` yazın ve Enter'a basın
- VEYA `cmd` yazın ve Enter'a basın

### Adım 2: Proje Klasörüne Gidin

```bash
cd "C:\sinir aglari"
```

### Adım 3: Paketleri Kontrol Edin (İlk Defa İse)

```bash
python -c "import torch; print('PyTorch hazır!')"
```

Eğer hata alırsanız:
```bash
pip install -r requirements.txt
```

### Adım 4: Modeli Eğitin

```bash
python train.py
```

---

## Ne Göreceksiniz?

### Terminal Çıktısı:

```
Using device: cpu
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
  ✓ Best model saved!
```

### Oluşturulan Dosyalar:

1. **`models/brain_tumor_cnn.pth`**
   - Eğitilmiş model dosyası
   - Bu dosya ile tahmin yapabilirsiniz

2. **`training_history.png`**
   - Eğitim ve validation loss grafikleri
   - Eğitim ve validation accuracy grafikleri
   - Herhangi bir resim görüntüleyici ile açabilirsiniz

---

## Tahmin Yapma (Model Eğitildikten Sonra)

### Yöntem 1: Otomatik Script

1. **`TAHMIN_YAP.bat`** dosyasına çift tıklayın
2. Görüntü yolunu girin (örn: `dataset\yes\Y1.jpg`)
3. Sonuçları görün

### Yöntem 2: Terminal Komutu

```bash
python predict.py --image dataset/yes/Y1.jpg
```

**Çıktı:**
```
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

---

## ⏱️ Eğitim Süresi

- **CPU**: 1-3 saat (30 epoch)
- **GPU**: 10-30 dakika (30 epoch)

**İlk Test İçin:** Epoch sayısını azaltabilirsiniz:
- `train.py` dosyasını açın
- 224. satırda: `'num_epochs': 30,` → `'num_epochs': 5,` yapın
- Hızlı test için 5 epoch yeterli

---

## Proje Klasör Yapısı

```
C:\sinir aglari\
├── dataset/                    # Veri seti (buraya dokunmayın)
├── models/                     # Eğitilmiş modeller 
│   └── brain_tumor_cnn.pth
├── BASLAT.bat                  #  Eğitimi başlatmak için
├── TAHMIN_YAP.bat              #  Tahmin yapmak için
├── train.py                    # Eğitim scripti
├── predict.py                   # Tahmin scripti
├── model.py                    # CNN modeli
├── data_loader.py              # Veri yükleme
└── requirements.txt            # Gerekli paketler
```

---

## Sık Sorulan Sorular

### Q: "ModuleNotFoundError: No module named 'torch'" hatası alıyorum

**Çözüm:**
```bash
pip install -r requirements.txt
```

### Q: Eğitim çok uzun sürüyor

**Çözüm:** 
- `train.py` dosyasında `'num_epochs': 5` yapın (hızlı test için)
- Veya GPU kullanın (CUDA kurulu olmalı)

### Q: Model dosyası nerede?

**Cevap:** `models/brain_tumor_cnn.pth` klasöründe

### Q: Grafikleri nasıl görürüm?

**Cevap:** `training_history.png` dosyasını herhangi bir resim görüntüleyici ile açın

---

## Özet - Hızlı Komutlar

```bash
# 1. Proje klasörüne git
cd "C:\sinir aglari"

# 2. Modeli eğit
python train.py

# 3. Tahmin yap (eğitimden sonra)
python predict.py --image dataset/yes/Y1.jpg
```

**VEYA sadece:**
- `BASLAT.bat` → Eğitimi başlat
- `TAHMIN_YAP.bat` → Tahmin yap

---

**Sorun mu var?** `KULLANIM.md` dosyasına bakın veya hata mesajını kontrol edin.







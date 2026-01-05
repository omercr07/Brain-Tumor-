# 🧠 Beyin Tümörü Tespiti - CNN Projesi

Bu proje, PyTorch kullanılarak geliştirilmiş bir **Convolutional Neural Network (CNN)** modeli ile beyin tümörü tespiti yapmaktadır. Model, MRI görüntülerinden tümör varlığını tespit etmek için eğitilmiştir.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Model Mimarisi](#model-mimarisi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Görselleştirmeler](#görselleştirmeler)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)
- [Gereksinimler](#gereksinimler)

## 🎯 Proje Hakkında

Bu proje, derin öğrenme teknikleri kullanarak beyin MRI görüntülerinde tümör tespiti yapan bir CNN modeli içermektedir. Model, binary classification (ikili sınıflandırma) yaparak görüntüleri "Tümör Var" veya "Tümör Yok" olarak sınıflandırır.

### Özellikler

- ✅ PyTorch ile geliştirilmiş modern CNN mimarisi
- ✅ Batch Normalization ve Dropout ile regularizasyon
- ✅ Data augmentation ile model performansının artırılması
- ✅ Eğitim, validasyon ve test seti ayrımı
- ✅ **Detaylı görselleştirmeler (Türkçe etiketlerle)**
- ✅ **Kaggle benzeri profesyonel çıktılar**
- ✅ Tek görüntü tahmini için hazır script
- ✅ Otomatik görselleştirme oluşturma

## 🏗️ Model Mimarisi

Model, aşağıdaki katman yapısına sahiptir:

### CNN Katmanları

1. **Convolutional Block 1**
   - Conv2d: 3 → 32 kanal, 3x3 kernel
   - Batch Normalization
   - ReLU aktivasyon
   - Max Pooling (2x2)

2. **Convolutional Block 2**
   - Conv2d: 32 → 64 kanal, 3x3 kernel
   - Batch Normalization
   - ReLU aktivasyon
   - Max Pooling (2x2)

3. **Convolutional Block 3**
   - Conv2d: 64 → 128 kanal, 3x3 kernel
   - Batch Normalization
   - ReLU aktivasyon
   - Max Pooling (2x2)

4. **Convolutional Block 4**
   - Conv2d: 128 → 256 kanal, 3x3 kernel
   - Batch Normalization
   - ReLU aktivasyon
   - Max Pooling (2x2)

5. **Fully Connected Layers**
   - Linear: 256×14×14 → 512
   - Dropout (0.5)
   - Linear: 512 → 128
   - Dropout (0.5)
   - Linear: 128 → 2 (sınıf sayısı)

### Model Özellikleri

- **Input Size**: 224×224×3 (RGB görüntüler)
- **Output**: 2 sınıf (No Tumor / Tumor)
- **Toplam Parametre**: ~15-20M (yaklaşık)
- **Regularizasyon**: BatchNorm + Dropout

## 🚀 Kurulum

### 1. Conda Environment Oluşturma

```bash
# Conda environment oluştur
conda env create -f environment.yml

# Environment'ı aktifleştir
conda activate brain_tumor_cnn
```

### 2. Pip ile Kurulum (Alternatif)

```bash
# Python virtual environment oluştur (opsiyonel)
python -m venv venv

# Virtual environment'ı aktifleştir
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Gereksinimleri yükle
pip install -r requirements.txt
```

### 3. PyTorch Kurulumu (Manuel)

Eğer PyTorch'u manuel olarak kurmak isterseniz:

```bash
# CPU versiyonu
pip install torch torchvision

# CUDA destekli versiyon (GPU için)
# Windows:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Linux/Mac için PyTorch resmi sitesinden uygun komutu kullanın
```

## 💻 Kullanım

### Model Eğitimi

Modeli eğitmek için `train.py` scriptini çalıştırın:

```bash
python train.py
```

**VEYA otomatik script ile:**

```bash
# Windows'ta
BASLAT.bat
```

Eğitim sırasında:
- Model otomatik olarak `models/` klasörüne kaydedilir
- En iyi model (en yüksek validation accuracy) kaydedilir
- **Tüm görselleştirmeler otomatik olarak oluşturulur**
- Test seti üzerinde sonuçlar gösterilir

### Tek Görüntü Tahmini

Eğitilmiş model ile tek bir görüntü üzerinde tahmin yapmak için:

```bash
python predict.py --image path/to/image.jpg
```

Örnek:
```bash
python predict.py --image dataset/yes/Y1.jpg
```

**VEYA otomatik script ile:**

```bash
# Windows'ta
TAHMIN_YAP.bat
```

### Görselleştirmeleri Yeniden Oluşturma

Mevcut model ile görselleri yeniden oluşturmak için:

```bash
python create_visualizations.py
```

Bu script, eğitilmiş modeli kullanarak tüm görselleştirmeleri oluşturur.

### Gelişmiş Kullanım

```bash
# Özel model yolu ile tahmin
python predict.py --image image.jpg --model models/my_model.pth

# CPU kullanımı (GPU varsa bile)
python predict.py --image image.jpg --device cpu
```

## 📊 Veri Seti

Veri seti yapısı:

```
dataset/
├── yes/              # Tümör olan görüntüler
│   ├── Y1.jpg
│   ├── Y2.jpg
│   └── ...
├── no/               # Tümör olmayan görüntüler
│   ├── 1 no.jpg
│   ├── 2 no.jpg
│   └── ...
└── brain_tumor_dataset/
    ├── yes/
    └── no/
```

### Veri Seti İstatistikleri

- **Toplam Görüntü**: ~500+ görüntü
- **Sınıf Dağılımı**:
  - Tümör Var (yes): ~310 görüntü
  - Tümör Yok (no): ~196 görüntü
- **Format**: JPG, JPEG, PNG
- **Boyut**: Değişken (model 224×224'e resize eder)

### Veri Bölünmesi

- **Training Set**: %70
- **Validation Set**: %10 (training'in %10'u)
- **Test Set**: %20

## 📈 Görselleştirmeler

Eğitim tamamlandıktan sonra, proje ana klasöründe aşağıdaki görselleştirmeler otomatik olarak oluşturulur:

### 1. Eğitim Geçmişi (`training_history.png`)
- Eğitim ve doğrulama loss grafikleri
- Eğitim ve doğrulama accuracy grafikleri
- Türkçe etiketler ve açıklamalar

### 2. Karışıklık Matrisi (`confusion_matrix.png`)
- Renkli görsel tablo
- Doğru ve yanlış tahminlerin görselleştirilmesi
- Türkçe sınıf isimleri

### 3. ROC Eğrisi (`roc_curve.png`)
- ROC (Receiver Operating Characteristic) eğrisi
- AUC (Area Under Curve) değeri
- Model performansının görsel analizi

### 4. Precision-Recall Eğrisi (`precision_recall_curve.png`)
- Precision ve Recall arasındaki ilişki
- Ortalama Precision değeri
- Dengesiz veri setleri için önemli metrik

### 5. Sınıf Dağılımı (`class_distribution.png`)
- Veri setindeki sınıf dağılımı
- Görsel çubuk grafik
- Yüzde ve sayı bilgileri

### 6. Örnek Tahminler (`sample_predictions.png`)
- Test setinden örnek görüntüler
- Gerçek ve tahmin edilen sınıflar
- Güven skorları
- Doğru/yanlış tahminlerin renkli gösterimi

**Tüm görseller Türkçe etiketler ve açıklamalar içerir!**

## 📊 Sonuçlar

### Eğitim Metrikleri

Model eğitimi tamamlandıktan sonra aşağıdaki metrikler gösterilir:

- **Training Loss & Accuracy**: Her epoch için
- **Validation Loss & Accuracy**: Her epoch için
- **Test Accuracy**: Final model performansı
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: Sınıflandırma detayları
- **ROC AUC**: ROC eğrisi altındaki alan
- **Average Precision**: Precision-Recall eğrisi altındaki alan

### Beklenen Performans

- **Validation Accuracy**: %85-95 arası
- **Test Accuracy**: %70-85 arası
- **Training Time**: 
  - CPU'da: 1-3 saat (30 epoch)
  - GPU'da: ~10-30 dakika (30 epoch)

*Not: Gerçek sonuçlar veri seti ve eğitim parametrelerine bağlı olarak değişebilir.*

## 📁 Proje Yapısı

```
.
├── dataset/                      # Veri seti klasörü
│   ├── yes/
│   ├── no/
│   └── brain_tumor_dataset/
├── models/                       # Eğitilmiş modeller
│   └── brain_tumor_cnn.pth
├── model.py                      # CNN model tanımı
├── data_loader.py                # Veri yükleme ve preprocessing
├── train.py                      # Eğitim scripti
├── predict.py                    # Tahmin scripti
├── create_visualizations.py      # Görselleştirme oluşturma scripti
├── requirements.txt              # Python gereksinimleri
├── environment.yml               # Conda environment dosyası
├── BASLAT.bat                    # Otomatik eğitim scripti (Windows)
├── TAHMIN_YAP.bat                # Otomatik tahmin scripti (Windows)
│
├── training_history.png          # Eğitim grafikleri (otomatik oluşturulur)
├── confusion_matrix.png          # Karışıklık matrisi (otomatik oluşturulur)
├── roc_curve.png                 # ROC eğrisi (otomatik oluşturulur)
├── precision_recall_curve.png    # Precision-Recall eğrisi (otomatik oluşturulur)
├── class_distribution.png        # Sınıf dağılımı (otomatik oluşturulur)
└── sample_predictions.png        # Örnek tahminler (otomatik oluşturulur)
```

## 📦 Gereksinimler

### Python Paketleri

- **torch** >= 2.0.0: PyTorch deep learning framework
- **torchvision** >= 0.15.0: Görüntü işleme ve veri setleri
- **numpy** >= 1.24.0: Sayısal hesaplamalar
- **Pillow** >= 9.5.0: Görüntü işleme
- **matplotlib** >= 3.7.0: Görselleştirme
- **seaborn** >= 0.12.0: İleri düzey görselleştirme
- **scikit-learn** >= 1.3.0: Metrikler ve veri bölme
- **tqdm** >= 4.65.0: Progress bar

### Sistem Gereksinimleri

- **Python**: 3.8+
- **RAM**: En az 8GB (önerilen: 16GB)
- **GPU**: Opsiyonel ama önerilir (CUDA destekli)
- **Disk**: ~2GB (veri seti + model + görseller)

## 🔧 Yapılandırma

Eğitim parametrelerini `train.py` dosyasındaki `config` dictionary'sinden değiştirebilirsiniz:

```python
config = {
    'batch_size': 32,        # Batch boyutu
    'image_size': 224,        # Görüntü boyutu
    'num_epochs': 30,         # Epoch sayısı (hızlı test için 5 yapabilirsiniz)
    'learning_rate': 0.001,   # Öğrenme oranı
    'dropout_rate': 0.5,      # Dropout oranı
}
```

## 📝 Notlar

- Model eğitimi GPU'da çok daha hızlı olacaktır
- Eğitim sırasında en iyi model otomatik olarak kaydedilir
- **Tüm görselleştirmeler otomatik olarak oluşturulur**
- Veri seti dengesizse, class weights kullanılabilir
- Data augmentation model performansını artırır
- Görseller proje ana klasörüne kaydedilir (kolay erişim için)

## 🎨 Görselleştirme Özellikleri

- ✅ **Türkçe etiketler ve açıklamalar**
- ✅ **Profesyonel ve renkli grafikler**
- ✅ **Kaggle benzeri çıktılar**
- ✅ **Yüksek çözünürlüklü görseller (300 DPI)**
- ✅ **Detaylı metrik gösterimleri**

## 🤝 Katkıda Bulunma

Bu proje eğitim amaçlıdır. İyileştirmeler ve öneriler için issue açabilirsiniz.

## 📄 Lisans

Bu proje eğitim amaçlıdır.

## 🔗 GitHub Repository

[GitHub Repository Linki](https://github.com/yourusername/brain-tumor-cnn)

---

**Not**: Bu model sadece eğitim ve araştırma amaçlıdır. Tıbbi teşhis için kullanılmamalıdır.

**Son Güncelleme**: Tüm görselleştirmeler Türkçe etiketlerle güncellenmiştir. Kaggle benzeri profesyonel çıktılar eklendi.

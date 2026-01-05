# Beyin Tümörü Tespiti - CNN Projesi

Bu proje, PyTorch kullanılarak geliştirilmiş bir **Convolutional Neural Network (CNN)** modeli ile beyin tümörü tespiti yapmaktadır. Model, MRI görüntülerinden tümör varlığını tespit etmek için eğitilmiştir.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Model Mimarisi](#model-mimarisi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Görselleştirmeler](#görselleştirmeler)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)
- [Gereksinimler](#gereksinimler)

##  Proje Hakkında

Bu proje, derin öğrenme teknikleri kullanarak beyin MRI görüntülerinde tümör tespiti yapan bir CNN modeli içermektedir. Model, binary classification (ikili sınıflandırma) yaparak görüntüleri "Tümör Var" veya "Tümör Yok" olarak sınıflandırır.

*** ONEMLI NOT: PROJEYI CALISTIRMAK ICIN TAHMIN_YAP.bat DOSYASINI YONETICI OLARAK CALISTIRIP  "dataset\yes\Y1.jpg" KODUNU YAZARAK ISTEDIGINIZ GORSEL HAKKINDA TUMOR DURUMUNU VE GEREKLI TAVSIYLERI GOREBILIRSINIZ.
KODDA GECEN "\yes\Y1.jpg" KISMI DEGISKEN OLUP CALISMA YAPMAK ISTEDIGIMIZ GORSELIN YOLUNU (PATH) GIREREK GEREKLI SONUCLARA ULASABILIRSINIZ.

### Özellikler

-  PyTorch ile geliştirilmiş modern CNN mimarisi
-  Batch Normalization ve Dropout ile regularizasyon
-  Data augmentation ile model performansının artırılması
-  Eğitim, validasyon ve test seti ayrımı
-  **Detaylı görselleştirmeler (Türkçe etiketlerle)**
-  **Kaggle benzeri profesyonel çıktılar**
-  Tek görüntü tahmini için hazır script
-  Otomatik görselleştirme oluşturma

## Model Mimarisi

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

## Kurulum

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

## Kullanım

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

## Görselleştirmeler

### 1. Eğitim Geçmişi (`training_history.png`)
- Eğitim ve doğrulama loss grafikleri
- Eğitim ve doğrulama accuracy grafikleri
- Türkçe etiketler ve açıklamalar
<img width="4772" height="1769" alt="training_historyy" src="https://github.com/user-attachments/assets/210832df-2cdf-499d-83c1-3a4efe7b4ab3" />

### 2. Karışıklık Matrisi (`confusion_matrix.png`)
- Renkli görsel tablo
- Doğru ve yanlış tahminlerin görselleştirilmesi
- Türkçe sınıf isimleri
<img width="2811" height="2379" alt="confusionn_matrix" src="https://github.com/user-attachments/assets/8875b8bb-398b-40ff-a647-2737b3a1be1c" />

### 3. ROC Eğrisi (`roc_curve.png`)
- ROC (Receiver Operating Characteristic) eğrisi
- AUC (Area Under Curve) değeri
- Model performansının görsel analizi
<img width="2967" height="2365" alt="roc_curvee" src="https://github.com/user-attachments/assets/14586d3c-6c2e-4cab-a4d7-9a38ca80030a" />

### 4. Precision-Recall Eğrisi (`precision_recall_curve.png`)
- Precision ve Recall arasındaki ilişki
- Ortalama Precision değeri
- Dengesiz veri setleri için önemli metrik
<img width="2966" height="2364" alt="precision_recall_curve" src="https://github.com/user-attachments/assets/8188cfe8-4269-49cb-9004-7b03e4ece40c" />

### 5. Sınıf Dağılımı (`class_distribution.png`)
- Veri setindeki sınıf dağılımı
- Görsel çubuk grafik
- Yüzde ve sayı bilgileri
<img width="2966" height="1768" alt="class_distribution" src="https://github.com/user-attachments/assets/a4dda4cb-ce10-4c3c-8c24-e9b4c689afc0" />

### 6. Örnek Tahminler (`sample_predictions.png`)
- Test setinden örnek görüntüler
- Gerçek ve tahmin edilen sınıflar
- Güven skorları
- Doğru/yanlış tahminlerin renkli gösterimi
<img width="4568" height="2403" alt="sample_predictions" src="https://github.com/user-attachments/assets/eb23c714-6061-4e1a-859b-b72daa45a30b" />

##  Proje Yapısı

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

##  Gereksinimler

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

##  Yapılandırma

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

##  Notlar

- Model eğitimi GPU'da çok daha hızlı olacaktır
- Eğitim sırasında en iyi model otomatik olarak kaydedilir
- **Tüm görselleştirmeler otomatik olarak oluşturulur**
- Veri seti dengesizse, class weights kullanılabilir
- Data augmentation model performansını artırır
- Görseller proje ana klasörüne kaydedilir (kolay erişim için)

##  Görselleştirme Özellikleri

-  **Türkçe etiketler ve açıklamalar**
-  **Profesyonel ve renkli grafikler**
-  **Kaggle benzeri çıktılar**
-  **Yüksek çözünürlüklü görseller (300 DPI)**
-  **Detaylı metrik gösterimleri**

##  Katkıda Bulunma

Bu proje eğitim amaçlıdır. İyileştirmeler ve öneriler için issue açabilirsiniz.

## GitHub Repository

[GitHub Repository Linki](https://github.com/yourusername/brain-tumor-cnn)



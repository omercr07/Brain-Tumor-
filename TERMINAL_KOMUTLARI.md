# 🖥️ Terminal Komutları - Kısa Rehber

## 📍 Adım 1: Terminal Aç
- `Windows + R` → `powershell` yaz → Enter
- VEYA `Windows + R` → `cmd` yaz → Enter

## 📍 Adım 2: Proje Klasörüne Git
```bash
cd "C:\sinir aglari"
```

## 📍 Adım 3: Modeli Eğit
```bash
python train.py
```

**Bittiğinde:**
- `models/brain_tumor_cnn.pth` → Model dosyası
- `training_history.png` → Grafikler

## 📍 Adım 4: Tahmin Yap (Eğitimden Sonra)
```bash
python predict.py --image dataset/yes/Y1.jpg
```

---

## ⚡ Hızlı Komutlar

```bash
# 1. Klasöre git
cd "C:\sinir aglari"

# 2. Eğit
python train.py

# 3. Tahmin yap
python predict.py --image dataset/yes/Y1.jpg
```

---

## 🔧 Sorun Çıkarsa

**"ModuleNotFoundError" hatası:**
```bash
pip install -r requirements.txt
```

**"python bulunamadı" hatası:**
- Python kurulu değil, önce Python kurun





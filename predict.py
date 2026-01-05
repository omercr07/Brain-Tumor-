"""
Prediction Script for Brain Tumor Detection
Loads a trained model and makes predictions on single images
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

from model import get_model


def load_model(model_path, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to the saved model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {'num_classes': 2, 'dropout_rate': 0.5})
    
    model = get_model(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image_path, image_size=224):
    """
    Preprocess an image for prediction
    
    Args:
        image_path: Path to the image file
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"HATA: Goruntu yuklenirken hata olustu: {e}")
        sys.exit(1)


def predict_image(model, image_tensor, device='cpu'):
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run on
        
    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['No Tumor', 'Tumor']
    predicted_class = class_names[predicted.item()]
    confidence_percent = confidence.item() * 100
    
    return predicted_class, confidence_percent, probabilities[0].cpu().numpy()


def get_tumor_detected_advice():
    """
    Tumör tespit edildiginde gösterilecek tavsiyeler
    
    Returns:
        str: Tavsiyeler metni
    """
    advice = """
    UYARI: TUMOR TESPIT EDILDI - ACIL ONLEMLER
    
    ACIL YAPILMASI GEREKENLER:
    
    1. DERHAL DOKTOR RANDEVUSU ALIN
       - En kisa surede bir noroloji veya onkoloji uzmanina basvurun
       - Acil servise gitmeyi dusunun (ciddi semptomlar varsa)
    
    2. DETAYLI TIBBI TESTLER YAPTIRIN
       - Kontrastli MR cekimi
       - CT taramasi
       - Biyopsi (gerekirse)
       - Kan testleri
    
    3. UZMAN GORUSU ALIN
       - Noroloji uzmani
       - Onkoloji uzmani
       - Beyin cerrahi (gerekirse)
       - Ikinci gorus almayi dusunun
    
    TEDAVI VE YASAM TARZI TAVSIYELERI:
    
    4. STRES YONETIMI
       - Meditasyon ve nefes egzersizleri yapin
       - Psikolojik destek alin
       - Aile ve arkadas destegi onemlidir
    
    5. SAGLIKLI BESLENME
       - Antioksidan acisindan zengin besinler (yesil yaprakli sebzeler, meyveler)
       - Omega-3 yag asitleri (balik, ceviz)
       - Islenmis gidalardan kacinin
       - Bol su icin
    
    6. DUZENLI UYKU
       - Gunde 7-8 saat kaliteli uyku
       - Duzenli uyku saatleri
       - Uyku hijyeni
    
    7. ZARARLI ALISKANLIKLARI BIRAKIN
       - Sigara kullanmayin
       - Alkol tuketimini sinirlandirin veya birakin
       - Sagliksiz yasam tarzindan kacinin
    
    8. IKINCI GORUS ALIN
       - Farkli uzmanlardan gorus alin
       - Tedavi seceneklerini degerlendirin
       - Karar vermeden once arastirin
    
    ONEMLI UYARI:
    Bu sonuc sadece bir on degerlendirmedir. Kesin tani icin mutlaka 
    uzman doktor kontrolunden gecmeniz gerekmektedir. Bu model sadece 
    egitim amaclidir ve tibbi teshis yerine gecmez.
    """
    return advice


def get_no_tumor_advice():
    """
    Tumör tespit edilmediginde gösterilecek tavsiyeler
    
    Returns:
        str: Tavsiyeler metni
    """
    advice = """
    TUMOR TESPIT EDILMEDI - KORUNMA TAVSIYELERI
    
    SAGLIKLI KALMAK ICIN YAPILMASI GEREKENLER:
    
    1. DUZENLI KONTROL YAPTIRIN
       - Yilda en az 1 kez beyin MR cekimi
       - Duzenli saglik kontrolleri
       - Aile oykusu varsa daha sik kontrol
    
    2. SAGLIKLI BESLENME
       - Sebze ve meyve agirlikli beslenme
       - Omega-3 yag asitleri (balik, ceviz, keten tohumu)
       - Antioksidan acisindan zengin besinler
       - Islenmis gidalardan kacinin
       - Seker ve tuz tuketimini sinirlandirin
    
    3. DUZENLI EGZERSIZ
       - Haftada en az 150 dakika orta yogunlukta egzersiz
       - Yuruyus, yuzme, bisiklet gibi aktiviteler
       - Duzenli fiziksel aktivite
    
    4. STRES YONETIMI
       - Meditasyon ve yoga
       - Hobi edinme
       - Sosyal aktiviteler
       - Yeterli dinlenme
    
    5. YETERLI UYKU
       - Gunde 7-8 saat kaliteli uyku
       - Duzenli uyku saatleri
       - Uyku hijyeni (karanlik, sessiz ortam)
    
    6. ZARARLI ALISKANLIKLARDAN KACININ
       - Sigara kullanmayin
       - Alkol tuketimini sinirlandirin
       - Sagliksiz yasam tarzindan kacinin
    
    7. GUNES KORUMASI
       - Gunes kremi kullanin
       - Sapka ve gozluk kullanin
       - Ogle saatlerinde gunesten kacinin
    
    8. SAGLIKLI KILO
       - Vucut kitle indeksinizi kontrol edin
       - Duzenli kilo takibi
       - Saglikli beslenme ve egzersiz
    
    EK TAVSIYELER:
    
    - Duzenli su tuketimi (gunde 2-3 litre)
    - Vitamin ve mineral takviyeleri (doktor kontrolunde)
    - Cevresel toksinlerden kacinma
    - Pozitif dusunce ve mental saglik
    
    ONEMLI NOT:
    Bu sonuc sadece bir on degerlendirmedir. Saglikli kalmak icin 
    duzenli kontrollerinizi aksatmayin. Herhangi bir supheli durumda 
    mutlaka uzman doktora basvurun.
    """
    return advice


def main():
    parser = argparse.ArgumentParser(description='Predict brain tumor from MRI image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the image file')
    parser.add_argument('--model', type=str, default='models/brain_tumor_cnn.pth',
                       help='Path to the trained model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"HATA: Goruntu dosyasi bulunamadi: {args.image}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"HATA: Model dosyasi bulunamadi: {args.model}")
        print("Lutfen once modeli egitin: python train.py")
        sys.exit(1)
    
    # Device configuration
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Kullanilan cihaz: {device}")
    
    # Load model
    print(f"Model yukleniyor: {args.model}...")
    model = load_model(args.model, device)
    print("Model basariyla yuklendi!")
    
    # Preprocess image
    print(f"\nGoruntu isleniyor: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Make prediction
    print("Tahmin yapiliyor...")
    predicted_class, confidence, probabilities = predict_image(model, image_tensor, device)
    
    # Display results
    print("\n" + "=" * 60)
    print("TAHMIN SONUCLARI")
    print("=" * 60)
    
    # Türkçe sınıf isimleri
    if predicted_class == "Tumor":
        predicted_class_tr = "Tumör Var"
    else:
        predicted_class_tr = "Tumör Yok"
    
    print(f"Tahmin Edilen Sinif: {predicted_class_tr}")
    print(f"Guven Orani: {confidence:.2f}%")
    print("\nSinif Olasiliklari:")
    print(f"  Tumör Yok: {probabilities[0]*100:.2f}%")
    print(f"  Tumör Var: {probabilities[1]*100:.2f}%")
    print("=" * 60)
    
    # Display advice based on prediction
    print("\n" + "=" * 60)
    if predicted_class == "Tumor":
        print(get_tumor_detected_advice())
    else:
        print(get_no_tumor_advice())
    print("=" * 60)
    
    return predicted_class, confidence


if __name__ == "__main__":
    main()







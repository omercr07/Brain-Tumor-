"""
Mevcut model ile görselleştirmeleri oluştur
Eğitilmiş model varsa, görselleri oluşturur
"""

import torch
import torch.nn as nn
import os
from model import get_model
from data_loader import get_dataloaders
from train import (plot_confusion_matrix, plot_roc_curve, 
                   plot_precision_recall_curve, plot_class_distribution,
                   plot_sample_predictions, test_model)

def main():
    print("=" * 60)
    print("Görselleştirmeler Oluşturuluyor...")
    print("=" * 60)
    
    # Model dosyası kontrolü
    model_path = 'models/brain_tumor_cnn.pth'
    if not os.path.exists(model_path):
        print(f"\nHATA: Model dosyası bulunamadı: {model_path}")
        print("Önce modeli eğitmeniz gerekiyor: python train.py")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nKullanılan cihaz: {device}")
    
    # Model yükle
    print(f"\nModel yükleniyor: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {'num_classes': 2, 'dropout_rate': 0.5})
    
    model = get_model(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model yüklendi!")
    
    # Veri yükle
    print("\nVeri seti yükleniyor...")
    _, _, test_loader = get_dataloaders(
        'dataset',
        batch_size=32,
        image_size=224,
        num_workers=0,  # Windows için 0
        augment=False
    )
    
    # Test et ve görselleri oluştur
    criterion = nn.CrossEntropyLoss()
    
    print("\nTest seti üzerinde model test ediliyor...")
    test_loss, test_acc, test_preds, test_labels, test_report, test_probs = test_model(
        model, test_loader, criterion, device, return_probs=True
    )
    
    print(f"\nTest Doğruluğu: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Görselleri oluştur
    print("\n" + "=" * 60)
    print("Görselleştirmeler oluşturuluyor...")
    print("=" * 60)
    
    # Confusion Matrix
    print("\n1. Karışıklık Matrisi oluşturuluyor...")
    plot_confusion_matrix(test_labels, test_preds, 
                         class_names=['Tümör Yok', 'Tümör Var'],
                         save_path='confusion_matrix.png')
    
    # ROC Curve
    print("2. ROC Eğrisi oluşturuluyor...")
    plot_roc_curve(test_labels, test_probs,
                   save_path='roc_curve.png')
    
    # Precision-Recall Curve
    print("3. Precision-Recall Eğrisi oluşturuluyor...")
    plot_precision_recall_curve(test_labels, test_probs,
                               save_path='precision_recall_curve.png')
    
    # Class Distribution
    print("4. Sınıf Dağılımı grafiği oluşturuluyor...")
    plot_class_distribution(test_loader,
                           save_path='class_distribution.png')
    
    # Sample Predictions
    print("5. Örnek Tahminler oluşturuluyor...")
    plot_sample_predictions(model, test_loader, device,
                           class_names=['Tümör Yok', 'Tümör Var'],
                           num_samples=8,
                           save_path='sample_predictions.png')
    
    print("\n" + "=" * 60)
    print("Tüm görselleştirmeler başarıyla oluşturuldu!")
    print("=" * 60)
    print("\nOluşturulan dosyalar:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - class_distribution.png")
    print("  - sample_predictions.png")
    print("\nNot: training_history.png dosyası eğitim sırasında oluşturulur.")
    print("=" * 60)

if __name__ == "__main__":
    main()


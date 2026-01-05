"""
Training Script for Brain Tumor Detection CNN
Handles model training, validation, and saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_curve, auc, precision_recall_curve, average_precision_score)
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches

from model import get_model
from data_loader import get_dataloaders

# Türkçe font ayarı
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def test_model(model, test_loader, criterion, device, return_probs=False):
    """
    Test the model on test set
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on
        return_probs: Whether to return prediction probabilities
        
    Returns:
        tuple: (loss, accuracy, predictions, labels, classification_report, probabilities)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        print("\nTest seti üzerinde model test ediliyor...")
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities for ROC/PR curves
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_probs:
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    # Classification report
    class_names = ['No Tumor', 'Tumor']
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    if return_probs:
        return test_loss, test_acc, all_preds, all_labels, report, all_probs
    return test_loss, test_acc, all_preds, all_labels, report


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """
    Plot training history with Turkish labels
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot losses
    ax1.plot(train_losses, label='Eğitim Loss', marker='o', linewidth=2, markersize=8)
    ax1.plot(val_losses, label='Doğrulama Loss', marker='s', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch (Dönem)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (Kayıp)', fontsize=12, fontweight='bold')
    ax1.set_title('Eğitim ve Doğrulama Loss Grafiği', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'Bu grafik modelin eğitim sırasındaki kayıp değerlerini gösterir.\nDüşen loss değerleri modelin öğrendiğini gösterir.',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot accuracies
    ax2.plot(train_accs, label='Eğitim Doğruluğu', marker='o', linewidth=2, markersize=8, color='green')
    ax2.plot(val_accs, label='Doğrulama Doğruluğu', marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Epoch (Dönem)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Doğruluk (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Eğitim ve Doğrulama Doğruluk Grafiği', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, 'Bu grafik modelin doğruluk oranını gösterir.\nYükselen doğruluk değerleri modelin performansının arttığını gösterir.',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEğitim geçmişi kaydedildi: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix with Turkish labels
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Örnek Sayısı'},
                xticklabels=class_names, yticklabels=class_names, 
                linewidths=0.5, linecolor='gray', annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('Karışıklık Matrisi (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=13, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    
    # Açıklama ekle
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    plt.text(0.5, -0.15, f'Toplam Örnek: {total} | Genel Doğruluk: {accuracy*100:.2f}%',
             transform=plt.gca().transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Karışıklık matrisi kaydedildi: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """
    Plot ROC curve with Turkish labels
    
    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Eğrisi (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Tahmin (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)', fontsize=13, fontweight='bold')
    plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)', fontsize=13, fontweight='bold')
    plt.title('ROC Eğrisi (Receiver Operating Characteristic)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Açıklama ekle
    plt.text(0.6, 0.2, f'AUC Değeri: {roc_auc:.3f}\n\nAUC değeri 1.0\'a ne kadar yakınsa,\nmodel o kadar iyi performans gösterir.\n0.5 değeri rastgele tahmin anlamına gelir.',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC eğrisi kaydedildi: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_scores, save_path='precision_recall_curve.png'):
    """
    Plot Precision-Recall curve with Turkish labels
    
    Args:
        y_true: True labels
        y_scores: Prediction scores/probabilities
        save_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkblue', lw=3, 
             label=f'Precision-Recall Eğrisi (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Hassasiyet / Geri Çağırma)', fontsize=13, fontweight='bold')
    plt.ylabel('Precision (Kesinlik)', fontsize=13, fontweight='bold')
    plt.title('Precision-Recall Eğrisi', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Açıklama ekle
    plt.text(0.05, 0.15, f'Ortalama Precision: {avg_precision:.3f}\n\nPrecision: Doğru tahmin edilen pozitiflerin oranı\nRecall: Tüm pozitiflerin ne kadarının bulunduğu',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall eğrisi kaydedildi: {save_path}")
    plt.close()


def plot_class_distribution(data_loader, save_path='class_distribution.png'):
    """
    Plot class distribution with Turkish labels
    
    Args:
        data_loader: DataLoader to get labels from
        save_path: Path to save the plot
    """
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    
    unique, counts = np.unique(all_labels, return_counts=True)
    class_names = ['Tümör Yok', 'Tümör Var']
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff']
    bars = plt.bar(class_names, counts, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
    
    # Değerleri çubukların üzerine yaz
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(all_labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Sınıf Dağılımı (Class Distribution)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sınıf', fontsize=13, fontweight='bold')
    plt.ylabel('Örnek Sayısı', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Açıklama ekle
    plt.text(0.5, 0.95, f'Toplam Örnek Sayısı: {len(all_labels)}',
             transform=plt.gca().transAxes, fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sınıf dağılımı grafiği kaydedildi: {save_path}")
    plt.close()


def plot_sample_predictions(model, data_loader, device, class_names, num_samples=8, save_path='sample_predictions.png'):
    """
    Plot sample predictions with images and labels
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save the plot
    """
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    samples_collected = 0
    with torch.no_grad():
        for images, labels in data_loader:
            if samples_collected >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(images)):
                if samples_collected >= num_samples:
                    break
                
                ax = axes[samples_collected]
                
                # Görüntüyü göster (denormalize et)
                img = images[i].cpu()
                img = img.permute(1, 2, 0)
                img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img = torch.clamp(img, 0, 1)
                
                ax.imshow(img.numpy())
                ax.axis('off')
                
                # Tahmin bilgisi
                true_label = class_names[labels[i].item()]
                pred_label = class_names[predicted[i].item()]
                confidence = probabilities[i][predicted[i]].item() * 100
                
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(f'Gerçek: {true_label}\nTahmin: {pred_label}\nGüven: {confidence:.1f}%',
                           fontsize=10, fontweight='bold', color=color)
                
                samples_collected += 1
    
    plt.suptitle('Örnek Tahminler (Sample Predictions)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Örnek tahminler kaydedildi: {save_path}")
    plt.close()


def main():
    # Configuration
    config = {
        'data_dir': 'dataset',
        'batch_size': 32,
        'image_size': 224,
        'num_epochs': 5,  # Hızlı test için 5, tam eğitim için 30
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'num_classes': 2,
        'num_workers': 4,
        'save_dir': 'models',
        'model_name': 'brain_tumor_cnn.pth'
    }
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        augment=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(config['save_dir'], config['model_name'])
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            print(f"  [OK] Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(config['save_dir'], config['model_name']))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model with probabilities
    test_loss, test_acc, test_preds, test_labels, test_report, test_probs = test_model(
        model, test_loader, criterion, device, return_probs=True
    )
    
    print("\n" + "=" * 60)
    print("TEST SONUÇLARI")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['No Tumor', 'Tumor']))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              No Tumor  Tumor")
    print(f"Actual No Tumor   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Tumor      {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print("\n" + "=" * 60)
    print("Görselleştirmeler oluşturuluyor...")
    print("=" * 60)
    
    # Tüm görselleri proje ana klasörüne kaydet
    # Plot training history (Türkçe) - ana klasöre
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                         save_path='training_history.png')
    
    # Plot confusion matrix (Türkçe) - ana klasöre
    plot_confusion_matrix(test_labels, test_preds, 
                         class_names=['Tümör Yok', 'Tümör Var'],
                         save_path='confusion_matrix.png')
    
    # Plot ROC curve (Türkçe) - ana klasöre
    plot_roc_curve(test_labels, test_probs,
                   save_path='roc_curve.png')
    
    # Plot Precision-Recall curve (Türkçe) - ana klasöre
    plot_precision_recall_curve(test_labels, test_probs,
                               save_path='precision_recall_curve.png')
    
    # Plot class distribution (Türkçe) - ana klasöre
    plot_class_distribution(test_loader,
                           save_path='class_distribution.png')
    
    # Plot sample predictions (Türkçe) - ana klasöre
    plot_sample_predictions(model, test_loader, device,
                           class_names=['Tümör Yok', 'Tümör Var'],
                           num_samples=8,
                           save_path='sample_predictions.png')
    
    print("\n" + "=" * 60)
    print("Eğitim başarıyla tamamlandı!")
    print(f"Model kaydedildi: {os.path.join(config['save_dir'], config['model_name'])}")
    print(f"\nGörselleştirmeler proje ana klasörüne kaydedildi:")
    print(f"  - training_history.png (Eğitim geçmişi)")
    print(f"  - confusion_matrix.png (Karışıklık matrisi)")
    print(f"  - roc_curve.png (ROC eğrisi)")
    print(f"  - precision_recall_curve.png (Precision-Recall eğrisi)")
    print(f"  - class_distribution.png (Sınıf dağılımı)")
    print(f"  - sample_predictions.png (Örnek tahminler)")
    print("=" * 60)


if __name__ == "__main__":
    main()


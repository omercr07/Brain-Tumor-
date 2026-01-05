"""
Rapor Oluşturma Scripti
Eğitim sonuçlarını ve görselleştirmeleri içeren detaylı rapor oluşturur
"""

import os
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix


def generate_html_report(test_report, test_acc, test_loss, cm, config, 
                         training_time, best_val_acc, save_path='model_report.html'):
    """
    HTML formatında detaylı rapor oluştur
    
    Args:
        test_report: Classification report dictionary
        test_acc: Test accuracy
        test_loss: Test loss
        cm: Confusion matrix
        config: Training configuration
        training_time: Training time in minutes
        best_val_acc: Best validation accuracy
        save_path: Path to save the report
    """
    
    # Sınıf isimleri
    class_names = ['Tümör Yok', 'Tümör Var']
    
    # Confusion matrix değerleri
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Metrikler
    precision_no = test_report['No Tumor']['precision']
    recall_no = test_report['No Tumor']['recall']
    f1_no = test_report['No Tumor']['f1-score']
    
    precision_yes = test_report['Tumor']['precision']
    recall_yes = test_report['Tumor']['recall']
    f1_yes = test_report['Tumor']['f1-score']
    
    html_content = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Beyin Tümörü Tespiti - Model Raporu</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
            max-width: 400px;
            margin: 20px auto;
        }}
        .cm-cell {{
            padding: 15px;
            text-align: center;
            border: 2px solid #3498db;
            border-radius: 5px;
            font-weight: bold;
        }}
        .cm-header {{
            background-color: #3498db;
            color: white;
        }}
        .cm-tn {{ background-color: #2ecc71; color: white; }}
        .cm-fp {{ background-color: #e74c3c; color: white; }}
        .cm-fn {{ background-color: #f39c12; color: white; }}
        .cm-tp {{ background-color: #27ae60; color: white; }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .image-card .caption {{
            padding: 10px;
            background-color: #f8f9fa;
            text-align: center;
            font-weight: bold;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Beyin Tümörü Tespiti - CNN Model Raporu</h1>
        
        <div class="info-box">
            <strong>Rapor Tarihi:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}<br>
            <strong>Model:</strong> Convolutional Neural Network (CNN)<br>
            <strong>Veri Seti:</strong> Beyin MRI Görüntüleri
        </div>
        
        <h2>📊 Özet Metrikler</h2>
        <div class="summary">
            <div class="metric-card">
                <h3>Test Doğruluğu</h3>
                <div class="value">{test_acc:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>En İyi Validation Doğruluğu</h3>
                <div class="value">{best_val_acc:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Test Loss</h3>
                <div class="value">{test_loss:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Eğitim Süresi</h3>
                <div class="value">{training_time:.1f} dk</div>
            </div>
        </div>
        
        <h2>📈 Detaylı Performans Metrikleri</h2>
        <table>
            <thead>
                <tr>
                    <th>Sınıf</th>
                    <th>Precision (Kesinlik)</th>
                    <th>Recall (Hassasiyet)</th>
                    <th>F1-Score</th>
                    <th>Destek</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Tümör Yok</strong></td>
                    <td>{precision_no:.4f}</td>
                    <td>{recall_no:.4f}</td>
                    <td>{f1_no:.4f}</td>
                    <td>{int(test_report['No Tumor']['support'])}</td>
                </tr>
                <tr>
                    <td><strong>Tümör Var</strong></td>
                    <td>{precision_yes:.4f}</td>
                    <td>{recall_yes:.4f}</td>
                    <td>{f1_yes:.4f}</td>
                    <td>{int(test_report['Tumor']['support'])}</td>
                </tr>
                <tr style="background-color: #f8f9fa; font-weight: bold;">
                    <td><strong>Ortalama</strong></td>
                    <td>{test_report['macro avg']['precision']:.4f}</td>
                    <td>{test_report['macro avg']['recall']:.4f}</td>
                    <td>{test_report['macro avg']['f1-score']:.4f}</td>
                    <td>{int(test_report['macro avg']['support'])}</td>
                </tr>
            </tbody>
        </table>
        
        <h2>🔢 Karışıklık Matrisi (Confusion Matrix)</h2>
        <div class="confusion-matrix">
            <div class="cm-cell cm-header"></div>
            <div class="cm-cell cm-header">Tahmin: Tümör Yok</div>
            <div class="cm-cell cm-header">Tahmin: Tümör Var</div>
            
            <div class="cm-cell cm-header">Gerçek: Tümör Yok</div>
            <div class="cm-cell cm-tn">{tn}</div>
            <div class="cm-cell cm-fp">{fp}</div>
            
            <div class="cm-cell cm-header">Gerçek: Tümör Var</div>
            <div class="cm-cell cm-fn">{fn}</div>
            <div class="cm-cell cm-tp">{tp}</div>
        </div>
        
        <div class="info-box">
            <strong>Açıklama:</strong><br>
            • <strong style="color: #27ae60;">Yeşil (TN, TP):</strong> Doğru tahminler<br>
            • <strong style="color: #e74c3c;">Kırmızı (FP):</strong> Yanlış pozitif (Tümör yokken tümör var dedi)<br>
            • <strong style="color: #f39c12;">Turuncu (FN):</strong> Yanlış negatif (Tümör varken tümör yok dedi)
        </div>
        
        <h2>📸 Görselleştirmeler</h2>
        <div class="image-gallery">
            <div class="image-card">
                <img src="visualizations/training_history.png" alt="Eğitim Geçmişi">
                <div class="caption">Eğitim ve Doğrulama Grafikleri</div>
            </div>
            <div class="image-card">
                <img src="visualizations/confusion_matrix.png" alt="Karışıklık Matrisi">
                <div class="caption">Karışıklık Matrisi</div>
            </div>
            <div class="image-card">
                <img src="visualizations/roc_curve.png" alt="ROC Eğrisi">
                <div class="caption">ROC Eğrisi</div>
            </div>
            <div class="image-card">
                <img src="visualizations/precision_recall_curve.png" alt="Precision-Recall Eğrisi">
                <div class="caption">Precision-Recall Eğrisi</div>
            </div>
            <div class="image-card">
                <img src="visualizations/class_distribution.png" alt="Sınıf Dağılımı">
                <div class="caption">Sınıf Dağılımı</div>
            </div>
            <div class="image-card">
                <img src="visualizations/sample_predictions.png" alt="Örnek Tahminler">
                <div class="caption">Örnek Tahminler</div>
            </div>
        </div>
        
        <h2>⚙️ Model Konfigürasyonu</h2>
        <table>
            <tr>
                <th>Parametre</th>
                <th>Değer</th>
            </tr>
            <tr>
                <td>Batch Size</td>
                <td>{config['batch_size']}</td>
            </tr>
            <tr>
                <td>Image Size</td>
                <td>{config['image_size']}x{config['image_size']}</td>
            </tr>
            <tr>
                <td>Epoch Sayısı</td>
                <td>{config['num_epochs']}</td>
            </tr>
            <tr>
                <td>Learning Rate</td>
                <td>{config['learning_rate']}</td>
            </tr>
            <tr>
                <td>Dropout Rate</td>
                <td>{config['dropout_rate']}</td>
            </tr>
            <tr>
                <td>Sınıf Sayısı</td>
                <td>{config['num_classes']}</td>
            </tr>
        </table>
        
        <h2>📝 Sonuç ve Değerlendirme</h2>
        <div class="info-box">
            <p><strong>Model Performansı:</strong></p>
            <p>Model, test seti üzerinde <strong>{test_acc:.2f}%</strong> doğruluk oranı elde etmiştir. 
            En iyi validation doğruluğu <strong>{best_val_acc:.2f}%</strong> olarak kaydedilmiştir.</p>
            
            <p><strong>Güçlü Yönler:</strong></p>
            <ul>
                <li>Tümör tespitinde yüksek precision ({precision_yes:.2%})</li>
                <li>İyi recall değeri ({recall_yes:.2%})</li>
                <li>Dengeli F1-score ({f1_yes:.4f})</li>
            </ul>
            
            <p><strong>İyileştirme Önerileri:</strong></p>
            <ul>
                <li>Daha fazla veri ile eğitim yapılabilir</li>
                <li>Data augmentation teknikleri artırılabilir</li>
                <li>Model mimarisi optimize edilebilir</li>
                <li>Hyperparameter tuning yapılabilir</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Bu rapor otomatik olarak oluşturulmuştur.</p>
            <p>Beyin Tümörü Tespiti CNN Projesi - PyTorch</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML raporu kaydedildi: {save_path}")


if __name__ == "__main__":
    # Test için örnek kullanım
    print("Rapor oluşturma scripti hazır!")
    print("Bu script train.py tarafından otomatik olarak çağrılacak.")


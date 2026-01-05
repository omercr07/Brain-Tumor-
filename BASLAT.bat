@echo off
echo ========================================
echo Beyin Tumoru Tespiti - CNN Projesi
echo ========================================
echo.

REM Proje klasörüne git
cd /d "%~dp0"

echo [1/2] Gerekli paketler kontrol ediliyor...
python -c "import torch" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo PyTorch bulunamadi! Kurulum yapiliyor...
    echo.
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo HATA: Paketler kurulamadi!
        pause
        exit /b 1
    )
    echo.
    echo Paketler basariyla kuruldu!
    echo.
) else (
    echo Tum paketler hazir!
    echo.
)

REM Model dosyası var mı kontrol et
if exist "models\brain_tumor_cnn.pth" (
    echo [2/2] Model zaten egitilmis! Gorsellestirmeler olusturuluyor...
    echo.
    echo Mevcut model kullanilarak gorsellestirmeler olusturulacak.
    echo (Egitim yapilmayacak, sadece gorseller olusturulacak)
    echo.
    echo ========================================
    echo.
    
    python create_visualizations.py
    set VIZ_RESULT=%ERRORLEVEL%
    
    echo.
    echo ========================================
    echo.
    
    if %VIZ_RESULT% EQU 0 (
        echo Gorsellestirmeler basariyla olusturuldu!
        echo.
        echo Olusturulan dosyalar:
        if exist "training_history.png" echo    - training_history.png (Egitim grafikleri)
        if exist "confusion_matrix.png" echo    - confusion_matrix.png (Karisiklik matrisi)
        if exist "roc_curve.png" echo    - roc_curve.png (ROC egrisi)
        if exist "precision_recall_curve.png" echo    - precision_recall_curve.png (Precision-Recall egrisi)
        if exist "class_distribution.png" echo    - class_distribution.png (Sinif dagilimi)
        if exist "sample_predictions.png" echo    - sample_predictions.png (Ornek tahminler)
        echo.
        echo Not: training_history.png dosyasi sadece egitim sirasinda olusturulur.
        echo.
        echo Simdi tahmin yapmak icin:
        echo    python predict.py --image dataset\yes\Y1.jpg
    ) else (
        echo Gorsellestirmeler olusturulurken bir hata olustu!
        echo Lutfen yukaridaki hata mesajini kontrol edin.
    )
) else (
    echo [2/2] Model bulunamadi! Model egitimi baslatiliyor...
    echo.
    echo Egitim basladi. Lutfen bekleyin...
    echo (Bu islem 1-3 saat surebilir - CPU kullaniyorsaniz)
    echo.
    echo ========================================
    echo.
    
    python train.py
    set TRAIN_RESULT=%ERRORLEVEL%
    
    echo.
    echo ========================================
    echo.
    
    REM Model dosyası oluştu mu kontrol et
    if exist "models\brain_tumor_cnn.pth" (
        echo Egitim basariyla tamamlandi!
        echo.
        echo Olusturulan dosyalar:
        echo    - models\brain_tumor_cnn.pth (Egitilmis model)
        if exist "training_history.png" echo    - training_history.png (Egitim grafikleri)
        if exist "confusion_matrix.png" echo    - confusion_matrix.png (Karisiklik matrisi)
        if exist "roc_curve.png" echo    - roc_curve.png (ROC egrisi)
        if exist "precision_recall_curve.png" echo    - precision_recall_curve.png (Precision-Recall egrisi)
        if exist "class_distribution.png" echo    - class_distribution.png (Sinif dagilimi)
        if exist "sample_predictions.png" echo    - sample_predictions.png (Ornek tahminler)
        echo.
        echo Simdi tahmin yapmak icin:
        echo    python predict.py --image dataset\yes\Y1.jpg
    ) else (
        echo Egitim sirasinda bir hata olustu!
        echo Lutfen yukaridaki hata mesajini kontrol edin.
    )
)
echo.
pause

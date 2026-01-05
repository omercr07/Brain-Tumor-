@echo off
echo ========================================
echo Beyin Tumoru Tahmini
echo ========================================
echo.

REM Proje klasörüne git
cd /d "%~dp0"

REM Model dosyası var mı kontrol et
if not exist "models\brain_tumor_cnn.pth" (
    echo HATA: Egitilmis model bulunamadi!
    echo.
    echo Once modeli egitmeniz gerekiyor:
    echo    python train.py
    echo    veya
    echo    BASLAT.bat dosyasini calistirin
    echo.
    pause
    exit /b 1
)

echo Model bulundu: models\brain_tumor_cnn.pth
echo.

:loop
echo.
echo ========================================
echo YENI TAHMIN
echo ========================================
echo.
set /p IMAGE_PATH="Dogru format: dataset\yes\Y1.jpg - Goruntu yolunu girin (cikis icin 'cikis' yazin): "

REM Çıkış kontrolü
if /i "%IMAGE_PATH%"=="cikis" (
    echo.
    echo Programdan cikiliyor...
    goto end
)

if "%IMAGE_PATH%"=="" (
    echo.
    echo HATA: Goruntu yolu girilmedi!
    echo.
    goto loop
)

REM Gereksiz komut kısımlarını temizle
set IMAGE_PATH=%IMAGE_PATH:python predict.py --image =%
set IMAGE_PATH=%IMAGE_PATH:python =%
set IMAGE_PATH=%IMAGE_PATH:predict.py =%
set IMAGE_PATH=%IMAGE_PATH:--image =%

REM Baştaki ve sondaki boşlukları temizle
for /f "tokens=*" %%a in ("%IMAGE_PATH%") do set IMAGE_PATH=%%a

REM Forward slash'leri backslash'e çevir
set IMAGE_PATH=%IMAGE_PATH:/=\%

REM Dosya kontrolü - hem tırnaklı hem tırnaksız dene
if exist "%IMAGE_PATH%" goto file_exists
if exist "%~dp0%IMAGE_PATH%" (
    set IMAGE_PATH=%~dp0%IMAGE_PATH%
    goto file_exists
)

echo.
echo HATA: Goruntu dosyasi bulunamadi: %IMAGE_PATH%
echo.
echo Mevcut dizin: %CD%
echo.
echo Dogru format: dataset\yes\Y1.jpg
echo Dogru format: dataset\no\1 no.jpeg
echo.
goto loop

:file_exists
echo.
echo ========================================
echo.

REM Python'a forward slash ile gönder (Python forward slash kabul eder)
set PYTHON_PATH=%IMAGE_PATH:\=/% 
python predict.py --image "%PYTHON_PATH%"

echo.
echo ========================================
echo.
echo Tahmin tamamlandi. Baska bir goruntu test etmek icin devam edin.
echo.

REM Döngüye geri dön
goto loop

:end
echo.
echo Program kapatildi.
timeout /t 2 >nul

@echo off
echo ========================================
echo Beyin Tumoru Tespiti CNN Projesi
echo ========================================
echo.

REM Conda environment kontrolu
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo HATA: Conda bulunamadi!
    echo Lutfen Anaconda veya Miniconda kurun.
    pause
    exit /b 1
)

echo [1/3] Conda environment olusturuluyor...
call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo HATA: Environment olusturulamadi!
    pause
    exit /b 1
)

echo.
echo [2/3] Environment aktiflestiriliyor...
call conda activate brain_tumor_cnn
if %ERRORLEVEL% NEQ 0 (
    echo HATA: Environment aktiflestirilemedi!
    pause
    exit /b 1
)

echo.
echo [3/3] Model egitimi baslatiliyor...
echo.
python train.py

echo.
echo ========================================
echo Egitim tamamlandi!
echo ========================================
echo.
echo Model dosyasi: models\brain_tumor_cnn.pth
echo Grafik: training_history.png
echo.
pause


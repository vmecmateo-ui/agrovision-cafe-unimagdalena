@echo off
echo ============================================
echo  AgroVision Cafe - UNIMAGDALENA
echo ============================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado.
    echo Instala Python 3.10+ desde https://python.org
    pause
    exit /b 1
)

REM Instalar dependencias si no están
echo Verificando dependencias...
pip install -q -r requirements.txt

echo.
echo Iniciando AgroVision Cafe...
python app.py

pause

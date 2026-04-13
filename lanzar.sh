#!/bin/bash
echo "============================================"
echo " AgroVision Café - UNIMAGDALENA"
echo "============================================"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 no encontrado."
    echo "Instala Python 3.10+ desde https://python.org"
    exit 1
fi

echo "Verificando dependencias..."
pip3 install -q -r requirements.txt

echo ""
echo "Iniciando AgroVision Café..."
python3 app.py

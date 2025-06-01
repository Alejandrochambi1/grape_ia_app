#!/bin/bash

echo "Iniciando GrapeAI - Sistema de Detección de Enfermedades en Uvas"
echo "================================================================"

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt

# Ejecutar aplicación
echo ""
echo "Iniciando servidor Flask..."
echo "La aplicación estará disponible en: http://localhost:5000"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""
python app.py

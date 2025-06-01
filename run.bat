@echo off
echo Iniciando GrapeAI - Sistema de Detección de Enfermedades en Uvas
echo ================================================================

REM Verificar si existe el entorno virtual
if not exist "venv" (
    echo Creando entorno virtual...
    py -3.11 -m venv venv
)

REM Activar entorno virtual
echo Activando entorno virtual...
call .\venv\Scripts\activate

REM Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt

REM Ejecutar aplicación
echo.
echo Iniciando servidor Flask...
echo La aplicación estará disponible en: http://localhost:5000
echo.
echo Presiona Ctrl+C para detener el servidor
echo.
python app.py

pause

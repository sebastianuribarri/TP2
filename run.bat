@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno virtual. Ejecuta setup.bat primero.
    exit /b 1
)

".venv\Scripts\python.exe" app.py

@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo No se encontro el lanzador "py". Instala Python 3.10 o superior y vuelve a intentar.
    exit /b 1
)

py -3 -c "import sys; print('Python detectado:', sys.version.split()[0]); raise SystemExit(sys.version_info < (3, 10))"
if errorlevel 1 (
    echo Se necesita Python 3.10 o superior. Instala una version compatible y vuelve a intentar.
    exit /b 1
)

py -3 -m venv .venv
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Entorno listo. Para iniciar la aplicacion, ejecuta run.bat.

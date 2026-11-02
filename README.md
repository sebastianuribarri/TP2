# Agente inteligente de gestión financiera

Aplicación de escritorio web para consultar una base de datos financiera SQLite con un agente LangGraph. Incluye chat, comparación de modelos y exploración del esquema.

## Requisitos

- Windows y Python 3.10 o superior, con el lanzador `py`.
- Una clave de Groq para usar el backend Groq. El backend Ollama requiere Ollama instalado y en ejecución.

## Instalación en Windows

Desde esta carpeta, ejecuta una sola vez:

```bat
setup.bat
```

El comando crea `.venv` e instala las dependencias de `requirements.txt`. Para iniciar la aplicación:

```bat
run.bat
```

Si se agregan dependencias al proyecto, vuelve a ejecutar `setup.bat` para instalarlas.

También puedes instalar las dependencias manualmente con `py -m pip install -r requirements.txt`.

## Configuración de modelos

La aplicación inicia con Groq como backend. Antes de ejecutar `run.bat`, define tu clave en PowerShell:

```powershell
$env:GROQ_API_KEY = "tu_clave"
.\run.bat
```

También puedes ingresar la clave en el campo correspondiente de la interfaz. Para Ollama, instala e inicia Ollama, descarga el modelo que quieras usar (por ejemplo, `ollama pull qwen3:8b`) y selecciona ese backend en la aplicación.

Al iniciar, la aplicación crea `gastos_empresa.db` con datos de ejemplo si todavía no existe. La memoria del chat se conserva en `memoria_conversacion.sqlite`: sobrevive al botón para limpiar la pantalla y a los reinicios del programa. Limpiar la pantalla borra los mensajes visibles, pero no la memoria que usa el agente. Para empezar desde cero, elimina ese archivo con la aplicación cerrada.

El archivo SQL incluido, `expense-tracker-records-v4-sql-sqlite.sql`, es un conjunto de datos separado y no se carga automáticamente en la aplicación actual.

## Archivos principales

- `app.py`: aplicación Gradio, agente, evaluación y creación de la base de datos.
- `requirements.txt`: dependencias Python.
- `setup.bat` / `run.bat`: preparación del entorno e inicio en Windows.
- `expense-tracker-records-v4-sql-sqlite.sql`: conjunto SQL independiente para explorar o integrar más adelante.

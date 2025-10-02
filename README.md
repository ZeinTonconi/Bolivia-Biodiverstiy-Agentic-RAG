# Agentic RAG - Biodiversidad de Bolivia

Este proyecto implementa un sistema de **Retrieval Augmented Generation (RAG)** especializado en la biodiversidad de Bolivia.  
El sistema permite realizar consultas sobre flora y fauna utilizando como base de conocimiento diversas fuentes oficiales y académicas.  
Cuando la consulta no se encuentra cubierta en la base de conocimientos, el sistema recurre a un agente de búsqueda en la web utilizando **SerpApi**.

La arquitectura incluye:
- **Backend (API con FastAPI)**: servidor encargado de la interacción con el modelo y el enrutamiento de consultas.
- **Frontend (React + Vite)**: interfaz de usuario en forma de chat para interactuar con el sistema.
- **Router híbrido**: encargado de decidir si la consulta se responde con la base de conocimientos (Bio Agent) o mediante búsqueda externa (Web Agent).

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone git@github.com:ZeinTonconi/Bolivia-Biodiverstiy-Agentic-RAG.git
cd Agentic-RAG
```
### 2. Crear un entorno virtual con uv

Este proyecto utiliza uv para la gestión de dependencias.
```
uv venv
source .venv/bin/activate  # en Linux/Mac
.venv\Scripts\activate     # en Windows PowerShell
```
### 3. Instalar dependencias
```
uv sync
```
Esto instalará todas las librerías necesarias (FastAPI, LlamaIndex, CrewAI, HuggingFace, entre otras).

### 4. Configuración del archivo `.env`

El proyecto requiere un archivo `.env` en la carpeta raíz para almacenar las claves privadas necesarias. En el repositorio se incluye un archivo `example.env` como referencia.

Copiar el archivo de ejemplo y renombrarlo:
```bash
cp example.env .env
```
Abrir el archivo .env y reemplazar con tus credenciales reales:
```
OPENAI_API_KEY=tu_openai_api_key_aqui
SERPER_API_KEY=tu_serpapi_api_key_aqui
```
## Ejecucion del proyecto
### 1. Servidor API (FastAPI)

Desde la carpeta raíz del proyecto ejecutar:
```
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```
El servidor se expondrá en http://127.0.0.1:8000.
Endpoints principales:
* **POST /ask:** recibe una consulta del usuario y devuelve la respuesta generada por el sistema.
* **POST /route:** ejecuta únicamente la función de enrutamiento para verificar qué agente debería encargarse de la consulta.
* **GET /status:** retorna el estado del servidor y del modelo.
* **GET /health:** verifica si el servidor está activo.

### 2. Interfaz de usuario (React)
Dentro de la carpeta ui ejecutar:
```
npm install
npm run dev
```
La interfaz estará disponible en http://localhost:5173.

## Generación de embeddings

El repositorio no incluye la carpeta biodiversity_store, debido a que su tamaño superaba los 100 MB y GitHub no permite alojarla.

Por esta razón, en la primera ejecución será necesario generar nuevamente todos los embeddings de los documentos de la base de conocimientos.
Este proceso puede tomar un tiempo considerable, dependiendo de la capacidad del equipo.

El sistema puede utilizar GPU si está disponible y cumple con los requisitos (CUDA 7.0 o superior). En tal caso, la generación de embeddings será significativamente más rápida:
```python
device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else "cpu"
print(f"[INFO] Embeddings running on: {device}")
```

Si no se dispone de GPU compatible, la generación se realizará en CPU.
## Modelos utilizados

* **Modelo de lenguaje:** gpt-4o-mini (OpenAI).

* **Modelo de embeddings:** intfloat/multilingual-e5-base (HuggingFace).
## Consideraciones
* La primera ejecución del sistema puede tardar debido a la generación de embeddings.
* Para un funcionamiento estable se recomienda utilizar una API Key de OpenAI de pago, ya que el uso de una cuenta gratuita puede ocasionar errores por límite de consultas (RateLimitError).
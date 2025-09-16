# Demo LLM

Para instalar las dependencias:

Crea un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows usa `.venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta la aplicación:

```bash
streamlit run app.py
```

Asegúrate de tener la variable de entorno `TOKEN` configurada con tu clave de API de OpenAI.

Puedes acceder a la aplicación en tu navegador en `http://localhost:8501`.
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("TOKEN"),
)

system_prompt = """
Eres Berto, el chatbot de la UBE (Universidad Bolivariana del Ecuador).

Tu tarea es brindar información a los estudiantes sobre la universidad,
los horarios, las clases, los profesores y cualquier otra consulta relacionada con la vida universitaria.

Responde siempre de forma amigable y profesional, en Español.

En ningún caso ayudes a los estudiantes a hacer la tarea o los exámenes.
Solo sugiere donde pueden estudiar.
"""

prompts = {
  "informacion general": """
HORARIO:

8am - 10am: Matemáticas
10am - 12pm: Física
12pm - 2pm: Almuerzo
2pm - 4pm: Programación
4pm - 6pm: Historia
6pm - 8pm: Literatura
8pm - 10pm: Deporte
10pm - 8am: Descanso

Mañana hay examen de Lenguas Extranjeras a las 11 am.

Estamos en semana de Festival.

MENU DEL DÍA:

Lunes: Pollo a la plancha con arroz y ensalada.
Martes: Pasta con salsa de tomate y carne.
Miércoles: Sándwich de atún con papas fritas.
Jueves: Ensalada César con pollo.
Viernes: Pizza con variedad de ingredientes.

Hoy es MARTES 16/9/2025.

Estamos en Guayaquil, Ecuador.
La temperatura es de 28 grados centígrados.
""",
  "informacion de los contenidos de las asignaturas": """
MATERIA: Matemáticas
TEMAS:
1. Álgebra: Ecuaciones, Inecuaciones, Polinomios.
2. Geometría: Figuras, Ángulos, Teoremas.
3. Cálculo: Límites, Derivadas, Integrales.
4. Estadística: Media, Mediana, Moda, Desviación estándar.

MATERIA: Física
TEMAS:
1. Mecánica: Movimiento, Fuerzas, Energía.
2. Termodinámica: Calor, Temperatura, Leyes de la termodinámica
3. Óptica: Luz, Reflexión, Refracción.
4. Electricidad: Carga, Corriente, Circuitos.

MATERIA: Lenguas Extranjeras
TEMAS:
1. Gramática: Tiempos verbales, Estructura de oraciones.
2. Vocabulario: Palabras comunes, Expresiones idiomáticas.
3. Comprensión lectora: Textos, Artículos, Narrativas.
4. Conversación: Diálogos, Pronunciación, Fluidez.
""",

  "soporte técnico de la web de la universidad": """
SOPORTE TÉCNICO:
1. Problemas de inicio de sesión: Restablecer contraseña, Verificar correo electrónico.
2. Navegación del sitio: Menús, Búsqueda, Accesibilidad.
3. Inscripción en cursos: Selección de materias, Confirmación de inscripción.
4. Pago de matrículas: Métodos de pago, Confirmación de pago, Facturación.
5. Recursos en línea: Biblioteca digital, Material de estudio, Foros de discusión.
"""
}

classifier = """
Clasifica la pregunta del usuario en:

- informacion general
- informacion de los contenidos de las asignaturas
- soporte técnico de la web de la universidad
- charla casual

Ejemplos:
charla casual: ¿Cómo estás hoy?
informacion general: ¿Cuál es el horario de clases para hoy?
informacion de los contenidos de las asignaturas: ¿Qué temas se cubrirán en la clase de matemáticas?
soporte técnico de la web de la universidad: No puedo iniciar sesión en mi cuenta de la universidad.

Responde solo con la categoria correcta, en un objeto JSON con llave "category".
"""

def reply():
  messages = list(st.session_state.history)
  messages.append({"role": "system", "content": classifier})

  classification = client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct",
    messages=messages,
    response_format={"type": "json_object"},
  )

  category = json.loads(classification.choices[0].message.content)["category"]
  st.chat_message("system").write(f"**Clasificación de la pregunta:** {category}")

  extra = prompts.get(category)

  if extra:
    messages.append({"role": "system", "content": extra})

  completion = client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct",
    messages=st.session_state.history,
    stream=True,
  )

  response = ""

  for chunk in completion:
    delta = chunk.choices[0].delta.content

    if delta:
      response += delta
      yield delta

  st.session_state.history.append({"role": "assistant", "content": response})

if "history" not in st.session_state:
  st.session_state.history = [
    {"role": "system", "content": system_prompt}
  ]

for msg in st.session_state.history:
  if msg["role"] == "system":
    continue

  with st.chat_message(msg["role"]):
    st.write(msg["content"])

msg = st.chat_input("Entra tu mensaje")

if not msg:
  st.stop()

with st.chat_message("user"):
  st.session_state.history.append({"role": "user", "content": msg})
  st.write(msg)

with st.chat_message("assistant"):
  st.write(reply())

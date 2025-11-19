import tensorflow as tf
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GLOBAL_STATE = {}

# === CLASSES ===
CLASSES = [
    {"class": 0, "company": "3DFX"},
    {"class": 1, "company": "ATI"},
    {"class": 2, "company": "MATROX"},
    {"class": 3, "company": "NVIDIA"},
    {"class": 4, "company": "S3"},
    {"class": 5, "company": "TRIDENT"},
]

# === CONFIGURACIÓN ===
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
RUTA_MODELO = os.path.join(CURRENT_DIR, "_model/compile/model_graphic_cards_transfer_learning.h5") 
IMG_SIZE = (224, 224)
DEBUG = False
ASK_DETAILS = os.getenv("ASK_DETAILS", "False") == "True"
api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_TOP_K = 80
MAX_RESPONSE_WORDS = 200

if DEBUG:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
else:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# === LOAD MODEL ===
print("Cargando modelo...")
model = tf.keras.models.load_model(RUTA_MODELO)

feature_layer = model.layers[-2]
feature_extractor = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
client = OpenAI(api_key=api_key)

def reduce_top_k(embedding, k=EMBEDDING_TOP_K):
    idx = np.argsort(np.abs(embedding))[-k:]
    return embedding[idx].tolist()

def extract_features(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0]
    class_idx = int(np.argmax(pred))
    class_label = CLASSES[class_idx]["company"]

    embedding = feature_extractor.predict(x)[0]
    embedding_list = reduce_top_k(embedding)

    return class_label, embedding_list

def ask_openai(label, embedding, top_k=EMBEDDING_TOP_K):
    prompt = f"""
    Eres un experto en hardware de tarjetas gráficas.
    Tu tarea es identificar y describir tarjetas gráficas en base a un embedding y la marca de la tarjeta gráfica.

    Para clasificar la tarjeta gráfica ten en cuenta que:
        - La marca de la tarjeta gráfica es: {label}
        - La imagen ha sido procesada utilizando un modelo de transferencia basado en MobileNetV2(preentrenado en ImageNet) para extraer un embedding original de 256 dimensiones.
        - De ese embedding se han seleccionado las {top_k} activaciones de mayor magnitud. Es decir, este vector representa las activaciones más fuertes (features más relevantes) detectadas por MobileNetV2 en la imagen, pero no incluye todas las dimensiones originales.
        - Este embedding reducido no representa filtros concretos ni índices específicos, sino una selección de los valores más significativos de la representación visual interna.
        - El embedding reducido está delimitado por triple backticks ```{embedding}```.

    A partir de esta información sigue estos pasos:

    1. Acota el análisis a la marca proporcionada.
    2. Basándote en la marca y el embedding reducido, infiere características técnicas.
    3. Describe la tarjeta gráfica probable.
    4. Aporta contexto histórico y técnico sobre la marca.
    5. Propón siempre un modelo específico de tarjeta gráfica que coincida con la descripción, ejemplos:
        - "NVIDIA GeForce 256 32MB DDR"
        - "ATI Radeon 9700 Pro 128MB DDR"
        - "3DFX Voodoo5 5500 64MB DDR"

    Responde en español de manera detallada.
    Responde con lenguaje claro y conciso.
    Responde con menos de 200 caracteres.
    Responde "No se responderte a la pregunta." si la pregunta está fuera de contexto.

    """
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def classify_and_ask(img):
    if img is None:
        return None, "No image provided.", {"class_label": None, "embedding": None, "history": []}
    
    class_label, embedding = extract_features(img)
    analysis = ask_openai(class_label, embedding) if ASK_DETAILS else "OpenAI desactivado."

    system_prompt = f"""
    Eres un experto en hardware de tarjetas gráficas de ordenador.

    Tu tarea es ayudar a los usuarios a entender y analizar tarjetas gráficas basándote en la información proporcionada.

    Tus respuestas deben ser breves, técnicas y concisas.

    Todas las preguntas y respuestas deben estar relacionadas con tarjetas gráficas de ordenador.

    Responde respetando los siguientes puntos:
    - Utiliza un lenguaje claro y conciso.
    - Con un máximo de {MAX_RESPONSE_WORDS} caracteres.
    - Responde "No se responderte a la pregunta." si la pregunta está fuera de contexto.
    - siempre en español.
    """

    # Estado inicial para chatbot
    state = {
        "class_label": class_label,
        "embedding": embedding,
        "history": [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": analysis}
        ]
    }

    return class_label, analysis, state

def chatbot_reply(user_message, history, state):
    global GLOBAL_STATE
    if not isinstance(state, dict):
        # Autocorrección si Gradio lo corrompe
        state = GLOBAL_STATE

    # Añadir mensaje del usuario al historial interno
    state["history"].append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=state["history"],
        temperature=0
    )

    reply = response.choices[0].message.content.strip()

    state["history"].append({"role": "assistant", "content": reply})

    GLOBAL_STATE = state

    # NO tocar history (Gradio lo gestiona)
    return reply, history, state

# ============================
# GRADIO INTERFACE
# ============================

with gr.Blocks(title="Graphics Card Classifier + Chatbot", fill_height=True) as app:

    gr.Markdown("# 🕹️ Graphics Card Classifier + Chatbot")
    gr.Markdown("Upload an image, get an initial analysis, and chat about it.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Image")
            analyze_btn = gr.Button("Analyze Image")
        with gr.Column(scale=2):
            class_output = gr.Label(label="Brand Predicted")
            analysis_output = gr.Textbox(label="Initial Analysis", lines=8)
    
    state = gr.State()

    analyze_btn.click(
        fn=classify_and_ask,
        inputs=image_input,
        outputs=[class_output, analysis_output, state]
    )

    gr.Markdown("💬 Chat about this graphics card")

    gr.ChatInterface(
        fn=chatbot_reply,
        additional_inputs=[state],
        additional_outputs=[state],
        title="Ask me about Graphics Cards"
    )

app.launch()

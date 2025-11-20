import tensorflow as tf
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import easyocr as easyocr

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
api_key = os.getenv("OPENAI_API_KEY")
MAX_RESPONSE_WORDS = 200

# == Models
blip_processor, blip_model = None, None
trco_processor, trco_model = None, None

if DEBUG:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
else:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# === LOAD MODEL ===
print("Cargando modelo transfer learning de TensorFlow...")
model = tf.keras.models.load_model(RUTA_MODELO)
print("Modelo cargado correctamente.")

#== LOAD BLIP or TrOCR MODELS ===
def load_model_choice(vision_extraction_method):
    global blip_processor, blip_model, trco_processor, trco_model

    if vision_extraction_method == "BLIP":
        print("Cargando BLIP para descripción de imágenes...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP cargado correctamente.")
    elif vision_extraction_method == "TrOCR":
        print("Cargando TrOCR para reconocimiento de texto en imágenes...")
        trco_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        trco_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        print("TrOCR cargado correctamente.")
    else:
        print("No se seleccionó ningún modelo de visión.")
        blip_processor, blip_model = None, None
        trco_processor, trco_model = None, None


feature_layer = model.layers[-2]
feature_extractor = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
client = OpenAI(api_key=api_key)

# === Get embedding features ===
def embedding_to_stats(embedding):
    return {
        "mean": float(np.mean(embedding)),
        "std": float(np.std(embedding)),
        "max": float(np.max(embedding)),
        "min": float(np.min(embedding)),
        "positive_ratio": float(np.sum(embedding > 0) / len(embedding)),
        "sparsity": float(np.sum(np.abs(embedding) < 0.1) / len(embedding))
    }

# == Get embedding categories ==
def embedding_to_categories(embedding):
    high = np.sum(embedding > 0.5)
    medium = np.sum((embedding > 0.1) & (embedding <= 0.5))
    low = np.sum(embedding <= 0.1)
    return f"High activations: {high}, Medium: {medium}, Low: {low}"

## == Get embedding features summary ==
def embedding_to_features(embedding):
    stats = embedding_to_stats(embedding)
    categories = embedding_to_categories(embedding)
    
    return f"""
    - Estadísticas: media={stats['mean']:.2f}, desv={stats['std']:.2f}
    - Distribución: {categories}
    - Activación positiva: {stats['positive_ratio']*100:.1f}%
    """

def extract_features(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0]
    class_idx = int(np.argmax(pred))
    class_label = CLASSES[class_idx]["company"]

    top_3_indices = np.argsort(pred)[-3:][::-1]
    top_3_predictions = [
        {
            "company": CLASSES[i]["company"],
            "probability": float(pred[i])
        }
        for i in top_3_indices
    ]

    embedding = feature_extractor.predict(x)[0]
    embedding_features = embedding_to_features(embedding)

    return class_label, embedding_features, top_3_predictions

def reset_conversation():
    return None, "", {"class_label": None, "embedding_features": None, "history": []}

def ask_openai_using_text_caption(label, caption_text, top_3_predictions):
    prompt = f"""
    Eres un experto en hardware de tarjetas gráficas.
    Tu tarea es identificar el modelo y describir tarjetas gráficas basándote en la marca y la descripción visual de la imagen de la tarjeta gráfica proporcionada por un modelo de captioning.

    Para clasificar la tarjeta gráfica sigue estas instrucciones:
    1. Usa la marca de la tarjeta gráfica proporcionada, que es la siguiente: {label}.
    2. Ten en cuenta las probabilidades de las tres marcas más probables para aumentar la precisión:
        {top_3_predictions}
    3. Usa la descripción visual para inferir características técnicas y el modelo probable:
        "{caption_text}"

    Tu respuesta debe seguir estas reglas:
    - Responde con menos de 200 caracteres.
    - Responde en español de manera detallada.
    - Responde con un modelo específico, por ejemplo:
        - "NVIDIA GeForce 256 32MB DDR"
        - "ATI Radeon 9700 Pro 128MB DDR"
        - "3DFX Voodoo5 5500 64MB DDR"
    - Responde con lenguaje claro y conciso.
    - Responde "No se responderte a la pregunta." si la pregunta no está relacionada con tarjetas gráficas de ordenador.

    """
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

def ask_openai_using_embeddings(label, embedding_features, top_3_predictions):
    prompt = f"""
    Eres un experto en hardware de tarjetas gráficas.
    Tu tarea es identificar el modelo y describir tarjetas gráficas basándote en la marca y un resumen del embedding de la imagen de la tarjeta gráfica proporcionado 
    un modelo entrenado por transferencia de aprendizaje de ImageNetV2.
    Este embedding reducido contiene información relevante como estadísticas, distribución de activaciones y porcentaje de activación positiva.

    Para clasificar la tarjeta gráfica sigue estas instrucciones:
    1. Usa la marca de la tarjeta gráfica proporcionada, que es la siguiente: {label}.
    2. Ten en cuenta las probabilidades de las tres marcas más probables:
        {top_3_predictions}
    3. Usa el resumen del embedding de la imagen de la tarjeta gráfica para inferir características técnicas y el modelo probable:
        {embedding_features}

    Tu respuesta debe seguir estas reglas:
    - Responde con menos de 200 caracteres.
    - Responde en español de manera detallada.
    - Responde con un modelo específico, por ejemplo:
        - "NVIDIA GeForce 256 32MB DDR"
        - "ATI Radeon 9700 Pro 128MB DDR"
        - "3DFX Voodoo5 5500 64MB DDR"
    - Responde con lenguaje claro y conciso.
    - Responde "No se responderte a la pregunta." si la pregunta no está relacionada con tarjetas gráficas de ordenador.

    """
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

def ask_openai_using_text_extraction_ocr(label, caption_text, top_3_predictions):
    prompt = f"""
    Eres un experto en hardware de tarjetas gráficas.
    Tu tarea es identificar el modelo y describir tarjetas gráficas basándote en la marca y el texto extraído de la imagen de la tarjeta gráfica proporcionado por OCR.

    Para clasificar la tarjeta gráfica sigue estas instrucciones:
    1. Usa la marca de la tarjeta gráfica proporcionada, que es la siguiente: {label}.
    2. Ten en cuenta las probabilidades de las tres marcas más probables para aumentar la precisión:
        {top_3_predictions}
    2. Usa el texto extraído por OCR para inferir características técnicas y el modelo probable indicado por triple backticks:
        ```{caption_text}```

    Tu respuesta debe seguir estas reglas:
    - Responde con menos de 200 caracteres.
    - Responde en español de manera detallada.
    - Responde con un modelo específico, por ejemplo:
        - "NVIDIA GeForce 256 32MB DDR"
        - "ATI Radeon 9700 Pro 128MB DDR"
        - "3DFX Voodoo5 5500 64MB DDR"
    - Responde con lenguaje claro y conciso.
    - Responde "No se responderte a la pregunta." si la pregunta no está relacionada con tarjetas gráficas de ordenador.

    """
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

def split_gpu_image(img):
    w, h = img.size

    # 4 recortes principales
    zones = {
        "top": img.crop((0, 0, w, int(h * 0.25))),                  # 0%–25%
        "upper_mid": img.crop((0, int(h * 0.20), w, int(h * 0.45))), # 20%–45%
        "center": img.crop((0, int(h * 0.40), w, int(h * 0.70))),    # 40%–70%
        "sticker_zone": img.crop((int(w * 0.20), int(h * 0.10), 
                                   int(w * 0.80), int(h * 0.40)))    # 20–80% width; 10–40% height
    }

    return zones

def ocr_region(img, box):
    x1, y1, x2, y2 = box

    # 🚧 Validar zona mínima
    if (x2-x1) < 10 or (y2-y1) < 10:
        return ""

    region = img.crop((x1, y1, x2, y2)).convert("RGB")

    # Escalar x2 o x3 mejora mucho OCR
    w, h = region.size
    region = region.resize((w*2, h*2), Image.Resampling.LANCZOS)
    
    return region

def ocr_reader(img):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(img))
    texts = [res[1] for res in result]

    return " ".join(texts)

def classify_and_ask(img, vision_method):
    if img is None:
        return None, "No image provided.", {"class_label": None, "embedding": None, "history": []}
    
    # Clasificación modelo transfer learning
    class_label, embedding_features, top_3_indices = extract_features(img)

    system_prompt = ""
    analysis = ""

    # Evaluar método de visión
    if vision_method in ["BLIP", "TrOCR", "EasyOCR", "Embeddings"]:
        if vision_method == "Embeddings":
            analysis = ask_openai_using_embeddings(class_label, embedding_features, top_3_indices)
        elif vision_method == "EasyOCR":
            zones = split_gpu_image(img)
            ocr_texts = []
            for zone_name, zone_img in zones.items():
                text = ocr_reader(zone_img)
                if text:
                    ocr_texts.append(f"{zone_name}: {text}")
            analysis = ask_openai_using_text_extraction_ocr(class_label, ",".join(ocr_texts), top_3_indices)
        elif vision_method == "TrOCR":
            zones = split_gpu_image(img)
            load_model_choice("TrOCR")
            ocr_texts = []
            for zone_name, zone_img in zones.items():
                pixel_values = trco_processor(images=img, return_tensors="pt").pixel_values
                generated_ids = trco_model.generate(pixel_values)
                text = trco_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if text:
                    ocr_texts.append(f"{zone_name}: {text}")
            analysis = ask_openai_using_text_extraction_ocr(class_label, ",".join(ocr_texts), top_3_indices)
        elif vision_method == "BLIP":
            load_model_choice("BLIP")
            inputs = blip_processor(images=img, return_tensors="pt")
            out = blip_model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            analysis = ask_openai_using_text_caption(class_label, caption, top_3_indices)
        else:
            analysis = "Not implemented yet."
            
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
    else:
        analysis = "LLM analysis not requested."

    # Estado inicial para chatbot
    state = {
        "class_label": class_label,
        "embedding_features": embedding_features,
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

def update_vision_method_interactivity(use_llm):
    """Habilita vision_method solo si use_llm está activado"""
    if use_llm:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False, value="None")

def validate_and_classify(img, use_llm, vision_method):
    """Valida que se cumplan las condiciones antes de analizar"""
    if img is None:
        return None, "❌ No image provided.", "", {"class_label": None, "embedding_features": None, "history": []}
    
    if use_llm and vision_method == "None":
        return None, "❌ Please select a Vision Extraction Method when using LLM Analysis.", "", {"class_label": None, "embedding_features": None, "history": []}
    
    # Si todo está bien, proceder con el análisis
    return classify_and_ask(img, vision_method)

with gr.Blocks(title="Graphics Card Classifier + Chatbot", fill_height=True) as app:

    gr.Markdown("# 🕹️ Graphics Card Classifier + Chatbot")
    gr.Markdown("Upload an image, get an initial analysis, and chat about it.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Image")
            
            use_llm = gr.Checkbox(
                label="Use LLM Analysis", 
                value=False,
                info="Generate detailed analysis with OpenAI"
            )

            vision_method = gr.Radio(
                choices=["None", "BLIP", "TrOCR", "EasyOCR", "Embeddings"],
                value="None",
                label="Vision Extraction Method",
                info="BLIP: Visual description (captioning) | TrOCR/EasyOCR: Text extraction | Embeddings: Feature summary",
                interactive=False
            )

            use_llm.change(
                fn=update_vision_method_interactivity,
                inputs=[use_llm],
                outputs=[vision_method]
            )

            analyze_btn = gr.Button("Analyze Image")
        with gr.Column(scale=2):
            class_output = gr.Label(label="Brand Predicted")
            analysis_output = gr.Textbox(label="Initial Analysis", lines=6)
    
    state = gr.State()

    image_input.clear(
        fn=reset_conversation,
        inputs=None,
        outputs=[class_output, analysis_output, state]
    )

    analyze_btn.click(
        fn=validate_and_classify,
        inputs=[image_input, use_llm, vision_method],
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
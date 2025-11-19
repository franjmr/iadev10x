import tensorflow as tf
import numpy as np
from PIL import Image
import os
from openai import OpenAI
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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
EMBEDDING_TOP_K = 40

if DEBUG:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Mostrar todos los mensajes de registro
else:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Mostrar mensajes de advertencia y errores
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ocultar mensajes informativos y de advertencia

# === CARGAR MODELO ===
print("Cargando modelo...")
model = tf.keras.models.load_model(RUTA_MODELO)

if DEBUG:
    model.summary()
    print("Output model:")
    print(model.output_shape)

# === PREPROCESAR IMAGEN ===
cards = [
    {"class": 0, "path": os.path.join(CURRENT_DIR, "_assets/card_3dfx.jpg")},
    {"class": 1, "path": os.path.join(CURRENT_DIR, "_assets/card_ati.jpg")},
    {"class": 2, "path": os.path.join(CURRENT_DIR, "_assets/card_matrox.jpg")},
    {"class": 3, "path": os.path.join(CURRENT_DIR, "_assets/card_nvidia.jpg")},
    {"class": 4, "path": os.path.join(CURRENT_DIR, "_assets/card_s3.jpg")},
    {"class": 5, "path": os.path.join(CURRENT_DIR, "_assets/card_trident.jpg")},
]

# Extraer vector de características
feature_layer = model.layers[-2]
feature_extractor = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)

def ask_openai(label, embedding, top_k=EMBEDDING_TOP_K):
    prompt = f"""
    Eres un experto en hardware de tarjetas gráficas.
    Tu tarea es identificar y describir tarjetas gráficas en base a un embedding y la marca de la tarjeta gráfica.

    Ten en cuenta que:
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
    5. Propón siempre un modelo específico de tarjeta gráfica que coincida con la descripción.

    Responde en español de manera detallada.
    Responde con lenguaje claro y conciso.
    Responde con menos de 200 caracteres.

    """
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def reduce_top_k(embedding, k=EMBEDDING_TOP_K):
    idx = np.argsort(np.abs(embedding))[-k:]  # los valores más grandes
    return embedding[idx].tolist()

def extract_features(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # clase predicha
    pred = model.predict(x)[0]
    class_idx = np.argmax(pred)
    class_label = CLASSES[class_idx]["company"]

    # embedding interno
    embedding = feature_extractor.predict(x)[0]
    embedding_list = reduce_top_k(embedding)

    return class_label, embedding_list

def predict_cards(cards):
    for card in cards:
        print("\n=== Procesando tarjeta gráfica ===")
        print(f": Clase {card['class']} - Company {CLASSES[card['class']]['company']} - Image Path: {card['path']}")
        
        img = Image.open(card["path"])
        class_label, embedding = extract_features(img)

        print(f"Predicción del modelo: {class_label}")

        if ASK_DETAILS:
            print("Consultando a OpenAI para más detalles...")
            details = ask_openai(class_label, embedding, top_k=EMBEDDING_TOP_K)
            print(f"Detalles de OpenAI:\n{details}")

        print("\n=============================\n")

predict_cards(cards)
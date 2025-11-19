import tensorflow as tf
import numpy as np
from PIL import Image
import os

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
RUTA_MODELO = "./_model/compile/model_graphic_cards_transfer_learning.h5" 
IMG_SIZE = (224, 224)
NORMALIZAR = True
DEBUG = False

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
    {"class": 0, "path": "./_assets/card_3dfx.jpg"},
    {"class": 1, "path": "./_assets/card_ati.jpg"},
    {"class": 2, "path": "./_assets/card_matrox.jpg"},
    {"class": 3, "path": "./_assets/card_nvidia.jpg"},
    {"class": 4, "path": "./_assets/card_s3.jpg"},
    {"class": 5, "path": "./_assets/card_trident.jpg"},
]

# Caculate accuracy over all cards
correct_predictions = 0
card_class_predicted = []

# Iterate cards to display their companies
for card in cards:
    print("\n=== Procesando tarjeta gráfica ===")
    print(f": {card['class']}, Image Path: {card['path']}")
    
    print("Cargando imagen...")
    img = Image.open(card['path']).convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.array(img, dtype=np.float32)

    if NORMALIZAR:
        x = x / 255.0

    x = np.expand_dims(x, axis=0)   # batch de 1

    # === PREDICCIÓN ===
    print("Realizando predicción...")
    pred = model.predict(x)

    print("\n=== Resultado ===")

    try:
        if DEBUG:
            print("Predicción en bruto:", pred)

        clase = np.argmax(pred, axis=-1)[0]
        print("Clase y nombre de la compañia predicha:", clase, CLASSES[clase]["company"])
        print("Clase y nombre de la compañía esperada:", card["class"], CLASSES[card["class"]]["company"])

        if clase == card["class"]:
            correct_predictions += 1
            card_class_predicted.append((card["class"], True))
        else:
            card_class_predicted.append((card["class"], False))

    except Exception as e:
        print(f"Error al obtener la clase predicha: {e}")

    print("\n=============================\n")

accuracy = correct_predictions / len(cards)
print(f"Precisión total del modelo: {accuracy * 100:.2f}%")

for card_class, predicted in card_class_predicted:
    status = "Correcta" if predicted else "Incorrecta"
    print(f"Clase {card_class} ({CLASSES[card_class]['company']}) > Predicción: {status}")
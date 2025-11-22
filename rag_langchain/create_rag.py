import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================
# 1. Cargar CSV limpio
# ============================

df = pd.read_csv(f"{CURRENT_DIR}/_assets/gpu_rag_clean_columns.csv")

# ============================
# 2. Cargar modelo embedding gratuito
# ============================

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ============================
# 3. Función para crear texto embeddable
# ============================

def create_embedding_text(row):
    def v(x):
        return str(x) if pd.notna(x) else "desconocido"

    return (
        f"{v(row['Brand'])} {v(row['Name'])}. "
        f"Arquitectura {v(row['Graphics Processor__Architecture'])}. "
        f"Año {v(row['Release_Year_Normalized'])}. "
        f"Memoria: {v(row['Memory__Memory Size'])} {v(row['Memory__Memory Type'])}, "
        f"bus {v(row['Memory__Memory Bus'])}. "
        f"Interfaz: {v(row['Graphics Card__Bus Interface'])}. "
        f"TDP {v(row['Board Design__TDP'])}. "
        f"{v(row['Render Config__Shading Units'])} shading units. "
        f"FP32: {v(row['Theoretical Performance__FP32 (float)'])}. "
        f"DirectX {v(row['Graphics Features__DirectX'])}, "
        f"OpenGL {v(row['Graphics Features__OpenGL'])}, "
        f"Shader Model {v(row['Graphics Features__Shader Model'])}. "
        f"Resolución recomendada: {v(row['Recommended Resolutions'])}."
    )

# ============================
# 4. Crear embeddings para Chroma
# ============================

# Inicializar Chroma con PersistentClient (nueva API)
client = chromadb.PersistentClient(path="./gpu_rag_chroma")

collection = client.get_or_create_collection(
    name="gpu_rag",
    metadata={"hnsw:space": "cosine"}
)

# Insertar los documentos
texts = []
ids = []
metadatas = []

print("Procesando GPUs y generando embeddings...")
for idx, row in df.iterrows():
    text = create_embedding_text(row)
    
    texts.append(text)
    ids.append(f"gpu_{idx}")
    
    # Convertir metadatos a tipos compatibles (solo str, int, float, bool)
    metadata = {}
    for key, value in row.to_dict().items():
        if pd.notna(value):
            if isinstance(value, (int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)
    
    metadatas.append(metadata)
    
    if (idx + 1) % 100 == 0:
        print(f"Procesadas {idx + 1} GPUs...")

print("Generando embeddings y añadiendo a la colección...")
embeddings = [model.encode(t).tolist() for t in texts]

collection.add(
    documents=texts, 
    embeddings=embeddings, 
    metadatas=metadatas, 
    ids=ids
)

print(f"✓ RAG cargado correctamente en Chroma con {len(texts)} GPUs.")

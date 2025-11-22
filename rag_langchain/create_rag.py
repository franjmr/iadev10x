import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
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
        f"Marca {v(row['Brand'])}. "
        f"Modelo {v(row['Name'])}. "
        f"Procesador Gráfico o GPU {v(row['Graphics Processor__GPU Name'])}. "
        f"Arquitectura {v(row['Graphics Processor__Architecture'])}. "
        f"Año {v(row['Release_Year_Normalized'])}. "
        f"Tamaño Memoria: {v(row['Memory__Memory Size'])} "
        f"Tipo Memoria: {v(row['Memory__Memory Type'])}, "
        f"Bus Memoria: {v(row['Memory__Memory Bus'])}. "
        f"Interfaz: {v(row['Graphics Card__Bus Interface'])}. "
        f"TDP {v(row['Board Design__TDP'])}. "
        f"{v(row['Render Config__Shading Units'])} shading units. "
        f"FP32: {v(row['Theoretical Performance__FP32 (float)'])}. "
        f"DirectX {v(row['Graphics Features__DirectX'])}, "
        f"OpenGL {v(row['Graphics Features__OpenGL'])}, "
        f"Shader Model {v(row['Graphics Features__Shader Model'])}. "
        f"Resolución recomendada: {v(row['Recommended Resolutions'])}."
    )


def clean_metadata(row):
    meta = {}
    for key, val in row.items():
        if pd.isna(val):
            continue
        if isinstance(val, (int, float, bool)):
            meta[key] = val
        else:
            meta[key] = str(val)
    return meta


# ============================
# 4. Crear embeddings para Chroma
# ============================

# Inicializar Chroma con PersistentClient (nueva API)
client = chromadb.PersistentClient(path=f"{CURRENT_DIR}/gpu_rag_chroma")

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
    metadatas.append(clean_metadata(row))
    
    if (idx + 1) % 100 == 0:
        print(f"Procesadas {idx + 1} GPUs...")

# Embeddings por lotes (batch)
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
embeddings = embeddings.tolist()

collection.add(
    documents=texts, 
    embeddings=embeddings, 
    metadatas=metadatas, 
    ids=ids
)

print(f"✓ RAG cargado correctamente en Chroma con {len(texts)} GPUs.")

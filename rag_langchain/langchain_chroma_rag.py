from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env")) 
API_KEY = os.getenv("OPENAI_API_KEY")

# ========= Embeddings locales =========
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ========= Cargar BD Chroma =========
vectorstore = Chroma(
    collection_name="gpu_rag",
    persist_directory=f"{CURRENT_DIR}/gpu_rag_chroma",
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ========= OpenAI LLM =========
llm = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

# ========= Prompt RAG =========
prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto en tarjetas gráficas retro y modernas.
Usa exclusivamente el siguiente contexto para responder.

CONTEXTO:
{context}

PREGUNTA:
{input}

RESPUESTA:
""")

# ========= Función para formatear documentos =========
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ========= CADENA RAG con LCEL =========
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========= BUCLE INTERACTIVO =========
print("=== RAG con LangChain 0.3.x y OpenAI listo ===")
print("Escribe 'salir' para terminar.\n")

while True:
    q = input("Tú: ").strip()

    if q.lower() in ["salir", "exit", "quit"]:
        print("Cerrando sesión.")
        break

    # Obtener documentos relevantes
    docs = retriever.invoke(q)
    
    # Invocar la cadena
    result = rag_chain.invoke(q)

    print("\n🧠 Respuesta:")
    print(result)

    print("\n📘 GPUs relevantes encontradas:")
    for doc in docs:
        meta = doc.metadata
        print(f"- {meta.get('Brand')} {meta.get('Name')} ({meta.get('Release_Year_Normalized')})")

    print("\n" + "="*50 + "\n")
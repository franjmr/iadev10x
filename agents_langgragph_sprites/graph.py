# graph.py
from langgraph.graph import StateGraph, END
from agents.generator import generate_sprite
from agents.validator import validate_sprite   # o quítalo si no quieres OpenAI
import os

# Estado compartido del grafo
class SpriteState(dict):
    prompt: str
    sprite_image: any
    validation: str | None
    sprite_b64: str | None


# ---- NODOS ----

def node_generate(state: SpriteState):
    print("🟦 Generando sprite local...")
    result = generate_sprite(state["prompt"])
    state["sprite_image"] = result["sprite_image"]
    return state

def node_validate(state: SpriteState):
    print("🟩 Validando sprite con OpenAI Vision...")
    result = validate_sprite(state["sprite_image"], state["prompt"])
    state["validation"] = result["validation"]
    state["sprite_b64"] = result["sprite_b64"]
    return state


# ---- UTILIDAD PARA GUARDAR ----

def save_sprite(image, path="output/sprite.png"):
    os.makedirs("output", exist_ok=True)
    image.save(path)
    print(f"📁 Sprite guardado en {path}")


# ---- DEFINICIÓN DEL GRAFO ----

workflow = StateGraph(SpriteState)

workflow.add_node("generate", node_generate)
workflow.add_node("validate", node_validate)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", END)

app = workflow.compile()


# ---- MAIN ----

if __name__ == "__main__":
    prompt = "a small green slime 16x16 style"

    result = app.invoke({"prompt": prompt})

    print("\n🧪 VALIDACIÓN:")
    print(result["validation"])

    save_sprite(result["sprite_image"])

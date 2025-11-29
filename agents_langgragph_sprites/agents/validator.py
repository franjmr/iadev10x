import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env")) 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def validate_sprite(pil_image, prompt):
    # Convertimos la imagen a base64
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    # Imagen en formato data-url (nuevo estándar OpenAI)
    img_url = f"data:image/png;base64,{img_b64}"

    # Texto del usuario
    query = (
        "Evalúa si este sprite es pixel-art y si cumple el prompt. "
        "Responde SOLO: VALIDO o NO VALIDO + explicación.\n\n"
        f"Prompt: {prompt}"
    )

    # Nueva estructura: input_text + input_image
    # response = client.responses.create(
    #     model="gpt-4o-mini",
    #     input=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "input_text", "text": query},
    #                 {"type": "input_image", "image_url": img_url}
    #             ]
    #         }
    #     ]
    # )

    # return {
    #     "validation": response.output_text,
    #     "sprite_b64": img_b64
    # }

    return {
        "validation": "VALIDO (simulación)",
        "sprite_b64": img_b64
    }
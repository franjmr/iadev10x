# agents/generator.py
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from dotenv import load_dotenv
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env")) 

print("➡️ Cargando modelo pixel-art (puede tardar unos segundos la primera vez)...")

# Cargar modelo una vez
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_auth_token=os.getenv("HF_TOKEN"),
    safety_checker=None
)
pipe.to("cpu")

def generate_sprite(prompt: str, size: int = 64):
    """
    Genera un sprite pixel-art 64x64 usando Diffusers (CPU).
    """
    full_prompt = (
        f"{prompt}. pixel art, retro game sprite, centered, "
        "simple shapes, thick outline, few colors, 16-bit style."
    )

    result = pipe(
        full_prompt,
        height=size,
        width=size,
        num_inference_steps=25
    )

    image: Image.Image = result.images[0]

    return {
        "sprite_image": image,
        "prompt": prompt
    }

# agents/generator.py
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image, ImageFilter
from dotenv import load_dotenv
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env")) 

print("➡️ Cargando modelo pixel-art (puede tardar unos segundos la primera vez)...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32,
    use_auth_token=os.getenv("HF_TOKEN")
)

pipe.to("cpu")  # CPU-only


def generate_sprite(prompt: str, size: int = 64):
    """
    Genera un sprite pixel-art simple usando SD-Turbo.
    Ultra rápido, ideal para iterar.
    """

    #megaman
    #init_image = Image.open(os.path.join(CURRENT_DIR, "..", "_assets", "megaman.jpg")).resize((256, 256))

    #shovel knight
    init_image = Image.open(os.path.join(CURRENT_DIR, "..", "_assets", "shovel.png")).resize((256, 256))
    base_prompt = f"retro video game sprite of {prompt}, tiny cartoon, clean silhouette, simple colors, 15 colors, no shading, flat colors, transparent background, no floor, centered character"

    # sd-turbo ignora guidance_scale (lo fuerza internamente)
    img = pipe(
        prompt=base_prompt,
        image=init_image,
        height=256,
        width=256,
        num_inference_steps=2, 
        strength=0.5, 
        guidance_scale=7.0
    ).images[0]

    # 1) Reducir para crear pixel-art REAL
    img_small = img.resize((size, size), Image.NEAREST)

    # 2) Cuantizar colores para aspecto retro auténtico
    img_quant = img_small.quantize(colors=6, method=Image.MEDIANCUT, dither=0)

    # 3) Reduce ruido ligero
    img_small = img_small.filter(ImageFilter.ModeFilter(size=3))

    # 4) Recortar bordes transparentes
    img = img.crop(img.getbbox())

    return {
        "sprite_image": img_quant.convert("RGB"),
        "prompt": prompt
    }
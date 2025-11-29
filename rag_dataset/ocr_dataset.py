from csv import reader
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import requests
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

blip_processor = None
blip_model = None

BASE_URL = "https://www.vgamuseum.info/index.php/cards"

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

def describe_image(img):
    global blip_processor, blip_model
    
    inputs = blip_processor(img, return_tensors="pt")
    output = blip_model.generate(**inputs, num_beams=5, max_new_tokens=200)

    return blip_processor.decode(output[0], skip_special_tokens=True)

def load_blip_model():
    global blip_processor, blip_model
    model_id = "Salesforce/blip-image-captioning-large"
    #model_id = "Salesforce/blip-image-captioning-base"
    #model_id = "Salesforce/blip-vqa-base"
    #model_id = "Salesforce/blip2-opt-2.7b"

    blip_processor = BlipProcessor.from_pretrained(model_id)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32  # CPU friendly
    )

def ocr_reader(img):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(img))
    texts = [res[1] for res in result]

    return " ".join(texts)

def auto_min_size(img):
    # Ajustar min_size según tamaño de imagen
    w, h = img.size
    min_size = max(3, round(h/200))  # mínimo 3, máximo h/200

    return min_size

def ocr_reader_extended(img):
    reader = easyocr.Reader(['en'], gpu=False)
    reader.readtext(
        np.array(img),
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.",
        min_size=auto_min_size(img)
    )
    result = reader.readtext(np.array(img))
    texts = [res[1] for res in result]

    return " ".join(texts)

def download_images_as_binary_rgb(cards):
    """
    Dada una lista de diccionarios con 'brand' y 'image_url',
    descarga cada imagen y crea nueva propiedad con el contenido binario.
    Devuelve cards actualizadas
    """

    for card in cards:
        brand = card.get("brand")
        url = card.get("image_url")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img_binary = BytesIO(response.content)
            img_rgb = Image.open(img_binary).convert("RGB")
            card["image_rgb"] = img_rgb
            print(f"Descargada: {brand} ({url})")

        except Exception as e:
            print(f"⚠️ No se pudo descargar {url}: {e}")

    return cards

def fetch_html_vgamuseum(url):
    """Descarga y devuelve el HTML de una URL del sitio vgamuseum.info"""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"❌ Error al acceder a {url}: {e}")
        return None

def extract_category_links_vgamuseum(base_url):
    """Extrae todos los enlaces a categorías principales desde /cards"""
    html = fetch_html_vgamuseum(base_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    fulltext_div = soup.find("div", class_="itemFullText")
    if not fulltext_div:
        print("⚠️ No se encontró div.itemFullText en la página principal.")
        return []

    category_links = []
    for li in fulltext_div.find_all("li"):
        a_tag = li.find("a", href=True)
        if a_tag:
            full_url = urljoin(base_url, a_tag["href"])
            category_links.append(full_url)
    print(f"🔗 Encontradas {len(category_links)} categorías.")
    return category_links

def extract_subcategory_links_vgamuseum(category_url):
    """
    Extrae las subpáginas de tarjetas (por modelo o serie) dentro de una categoría,
    gestionando la paginación localizada en 'ul.pagination-list a.pagenav'.
    """
    html = fetch_html_vgamuseum(category_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # 1️⃣ Obtener todas las URLs de paginación (incluyendo la página inicial)
    pagination_urls = [category_url]
    pagination_section = soup.select("ul.pagination-list a.pagenav")

    for a_tag in pagination_section:
        page_url = urljoin(category_url, a_tag["href"])
        if page_url not in pagination_urls:
            pagination_urls.append(page_url)

    print(f"  📁 {len(pagination_urls)} páginas de subcategorías encontradas en {category_url}")

    # 2️⃣ Iterar por todas las páginas (original + paginadas) y extraer subpáginas
    sub_links = []
    for page_url in pagination_urls:
        html_page = fetch_html_vgamuseum(page_url)
        if not html_page:
            continue

        soup_page = BeautifulSoup(html_page, "html.parser")

        for h3 in soup_page.find_all("h3", class_="catItemTitle"):
            a_tag = h3.find("a", href=True)
            if a_tag:
                full_url = urljoin(page_url, a_tag["href"])
                sub_links.append(full_url)

    print(f"  📁 {len(sub_links)} subcategorías encontradas en {category_url} (paginación incluida)")
    return sub_links


def extract_images_from_subpage_vgamuseum(sub_url):
    """
    Extrae todas las imágenes JPG dentro de div.itemFullText,
    descarta las que contengan 'driv' y devuelve también el título
    (h2.itemTitle dentro de div.itemHeader).
    """
    html = fetch_html_vgamuseum(sub_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # 🔹 Obtener el título de la subpágina
    title = None
    header_div = soup.find("div", class_="itemHeader")
    if header_div:
        title_tag = header_div.find("h2", class_="itemTitle")
        if title_tag:
            title = title_tag.get_text(strip=True)

    fulltext_divs = soup.find_all("div", class_="itemFullText")
    images = []

    for div in fulltext_divs:
        for a_tag in div.find_all("a", href=True):
            href = a_tag["href"]
            if href.lower().endswith(".jpg"):
                img_url = urljoin(sub_url, href)

                # Descartar imágenes relacionadas con drivers
                if "driv" in img_url.lower() or "drv" in img_url.lower() or "driver" in img_url.lower():
                    print(f"    ⚠️ Descartada imagen de drivers: {img_url}")
                    continue

                images.append({
                    "title": title or "Sin título",
                    "image_url": img_url
                })

    if images:
        print(f"    🖼️ {len(images)} imágenes válidas encontradas en {sub_url} — {title or 'Sin título'}")
    return images

def scrape_vgamuseum_cards(base_url):
    """Scraper principal para recorrer todo el árbol de categorías y subcategorías con paginación"""
    all_data = []

    categories = extract_category_links_vgamuseum(base_url)

    print(f"\n🔎 Recuperadas {len(categories)} categorías principales.")

    included_categories = [
        #"https://www.vgamuseum.info/index.php/cards/itemlist/category/31-s3-graphics",
        "https://www.vgamuseum.info/index.php/cards/itemlist/category/2-3dfx",
        #"https://www.vgamuseum.info/index.php/cards/itemlist/category/22-matrox",
        #"https://www.vgamuseum.info/index.php/cards/itemlist/category/9-ati-technologies-inc",
        #"https://www.vgamuseum.info/index.php/cards/itemlist/category/27-nvidia-corporation",
        #"https://www.vgamuseum.info/index.php/cards/itemlist/category/39-trident-microsystems-inc"
    ]

    for cat_url in categories:
        if cat_url not in included_categories:
            continue

        sub_links = extract_subcategory_links_vgamuseum(cat_url)
        for sub_url in sub_links:
            images = extract_images_from_subpage_vgamuseum(sub_url)

            category_from_url = cat_url.split("/")[-1]
            category = "-".join(category_from_url.split("-")[1:])

            for img_info in images:
                all_data.append({
                    "brand": category,
                    "brand_url": cat_url,
                    "subpage_url": sub_url,
                    "model": img_info["title"],
                    "image_url": img_info["image_url"]
                })
            time.sleep(1)  # evita sobrecargar el servidor

    return all_data

# Load BLIP model once
load_blip_model()

# Ejecutar el scraping completo (sin guardar CSV)
cards = scrape_vgamuseum_cards(BASE_URL)

# Limitar a las primeras 2 para pruebas
download_images_as_binary_rgb(cards[:2])

# Procesar las primeras 2 imágenes como ejemplo
for card in cards[:2]:
    title = card["model"]
    img = card.get("image_rgb")
    if img is None:
        print(f"⚠️ Imagen RGB no disponible para {title}")
        continue

    print(f"Procesando tarjeta: {title} con url {card['image_url']}")

    # compare OCR by zones vs full image

    # OCR en imagen completa
    print(f"\n🔍 Procesando imagen: {title}")
    full_img_timer = time.time()
    full_img_text = ocr_reader_extended(img)
    full_img_duration = time.time() - full_img_timer
    print(f"  📝 OCR imagen completa: {full_img_text} (tiempo: {full_img_duration:.2f}s)")

    # # OCR por zonas
    # zones_texts_timer = time.time()
    # zones = split_gpu_image(img)
    # zones_texts = {}
    # for zone_name, zone_img in zones.items():
    #     text = ocr_reader_extended(zone_img)
    #     zones_texts[zone_name] = text

    # combined_zone_text = " ".join(zones_texts.values())
    # zones_texts_duration = time.time() - zones_texts_timer
    # print(f"  📝 OCR zonas combinadas: {combined_zone_text} (tiempo: {zones_texts_duration:.2f}s)")

    # Descripción BLIP
    blip_timer = time.time()
    blip_description = describe_image(img)
    blip_duration = time.time() - blip_timer
    print(f"  🖼️ Descripción BLIP: {blip_description} (tiempo: {blip_duration:.2f}s)")
    print("=" * 50)
    
import pandas as pd
import os
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_COLS_FINAL = [
    "Brand",
    "Name",
    "Graphics Processor__GPU Name",
    "Graphics Processor__Architecture",
    "Release_Year_Normalized",
    "Memory__Memory Size",
    "Memory__Memory Type",
    "Memory__Memory Bus",
    "Graphics Card__Bus Interface",
    "Board Design__TDP",
    "Render Config__Shading Units",
    "Theoretical Performance__FP32 (float)",
    "Graphics Features__DirectX",
    "Graphics Features__OpenGL",
    "Graphics Features__Shader Model",
    "Recommended Resolutions"
]

def print_df_porcent_missing(df):
    # % de valores vacíos por columna
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    for column, pct in missing_pct.items():
        if column in EMBED_COLS_FINAL:
            if pct <= 20:
                #pintar linea en verde si es menor a 20% porque es hay que Mantener (útil en embeddings y metadata)
                print(f"\033[92mLa columna '{column}' tiene {pct:.2f}% de valores vacíos. Mantener.\033[0m")
            elif pct > 20 and pct < 50:
                #pintar linea en amarillo si es menor a 50% porque es hay que Mantener solo en metadata
                print(f"\033[93mLa columna '{column}' tiene {pct:.2f}% de valores vacíos. Evaluar.\033[0m")
            elif pct >= 50 and pct < 80:
                #pintar linea en naranja si es menor a 80% porque es hay que Eliminar de embeddings
                print(f"\033[33mLa columna '{column}' tiene {pct:.2f}% de valores vacíos. Eliminar de embeddings.\033[0m")
            else:
                #pintar linea en rojo si es mayor a 80% porque es hay que Eliminar
                print(f"\033[91mLa columna '{column}' tiene {pct:.2f}% de valores vacíos. Eliminar.\033[0m")

def extract_year_from_release_date(value):
    if pd.isna(value):
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None

def extract_year_from_source_file(value):
    if pd.isna(value):
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None

def get_release_year(row):
    y1 = extract_year_from_release_date(row.get("Graphics Card__Release Date"))
    if y1:
        return y1
    y2 = extract_year_from_source_file(row.get("source_file"))
    return y2

def normalize_shading_units(row):
    value = row.get("Render Config__Shading Units")
    shader_model = row.get("Graphics Features__Shader Model")

    # Caso 1: Si tiene shader units → mantener (pero como texto)
    if pd.notna(value):
        return str(value).strip()

    # Caso 2: GPUs sin Shader Model = fixed pipeline
    # Ejemplos: Voodoo, Savage, Matrox Gx00, TNT/Pro
    if pd.isna(shader_model):
        return "fixed-pipeline"

    # Caso 3: GPUs modernas sin dato exacto
    return "unknown"

def normalize_fp32(value):
    if pd.isna(value):
        return "unknown"
    try:
        return float(value)
    except:
        return "unknown"

def normalize_bus_interface(value):
    if pd.isna(value):
        return "unknown"

    s = str(value).strip().lower()

    # eliminar valores raros (a veces viene "2010" o cadenas vacías)
    if re.fullmatch(r"\d{4}", s):
        return "unknown"

    # normalizaciones básicas
    s = s.replace("agp pro", "agp")
    s = s.replace("agp 4x", "agp4x")
    s = s.replace("agp 8x", "agp8x")
    s = s.replace("agp 2x", "agp2x")

    # normalización PCI Express
    s = s.replace("pci express", "pcie")
    s = s.replace("pci-e", "pcie")
    s = s.replace("pci ex", "pcie")

    # formatear PCIe versiones
    s = re.sub(r"pcie\s*x?(\d+)", r"pcie x\1", s)

    # si tras limpiar queda vacío
    if s in ["", "-", "n/a", "none"]:
        return "unknown"

    return s

df = pd.read_csv(f"{CURRENT_DIR}/_assets/gpu_1986-2026.csv")

# Estadísticas descriptivas
print("Estadísticas iniciales de valores vacíos por columna:")
print_df_porcent_missing(df)
print("\n")

df["Render Config__Shading Units"] = df.apply(normalize_shading_units, axis=1)
df["Theoretical Performance__FP32 (float)"] = df["Theoretical Performance__FP32 (float)"].apply(normalize_fp32)
df["Graphics Card__Bus Interface"] = df["Graphics Card__Bus Interface"].apply(normalize_bus_interface)
df["Release_Year_Normalized"] = df.apply(get_release_year, axis=1)


# Estadísticas normalizadas
print("Estadísticas finales de valores vacíos por columna:")
print_df_porcent_missing(df)
print("\n")

# Guardar CSV para análisis manual
# Filtrar solo columnas existentes
cols_existentes = [c for c in EMBED_COLS_FINAL if c in df.columns]

df_clean = df[cols_existentes].copy()
df_clean.to_csv(f"{CURRENT_DIR}/_assets/gpu_rag_clean_columns.csv", index=False, encoding="utf-8")

print("CSV generado: gpu_rag_clean_columns.csv")
print(f"Columnas incluidas: {cols_existentes}")
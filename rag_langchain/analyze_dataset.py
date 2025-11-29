import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Directorio de trabajo
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar datos
df = pd.read_csv(os.path.join(CURRENT_DIR, '_assets', 'gpu_1986-2026.csv'))

# 1. LIMPIEZA INICIAL - MEJORADA
# Función para extraer año del source_file
def extract_year_from_source(source_file):
    """Extrae el año del nombre del archivo source_file"""
    if pd.isna(source_file):
        return None
    # Buscar patrón de 4 dígitos que represente un año (1900-2099)
    match = re.search(r'(19\d{2}|20\d{2})', str(source_file))
    if match:
        return int(match.group(1))
    return None

# Convertir Release Date a datetime
df['Graphics Card__Release Date'] = pd.to_datetime(df['Graphics Card__Release Date'], errors='coerce')
df['Year_from_Release'] = df['Graphics Card__Release Date'].dt.year

# Extraer año del source_file
df['Year_from_Source'] = df['source_file'].apply(extract_year_from_source)

# Usar Year_from_Source cuando Year_from_Release esté vacío
df['Year'] = df['Year_from_Release'].fillna(df['Year_from_Source'])

# Mostrar estadísticas de los años
print("="*60)
print("ESTADÍSTICAS DE AÑOS")
print("="*60)
print(f"Total de registros: {len(df)}")
print(f"Registros con fecha de lanzamiento: {df['Year_from_Release'].notna().sum()}")
print(f"Registros sin fecha pero con source_file: {(df['Year_from_Release'].isna() & df['Year_from_Source'].notna()).sum()}")
print(f"Registros sin año alguno: {df['Year'].isna().sum()}")
print(f"\nRango de años: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
print("="*60)

# 2. ANÁLISIS SUGERIDOS

# A) Evolución del tamaño de proceso
df['Process_nm'] = df['Graphics Processor__Process Size'].str.extract('(\d+)').astype(float)

# B) Evolución de transistores
df['Transistors_M'] = df['Graphics Processor__Transistors'].str.extract('(\d+)').astype(float)

# C) Evolución de TDP
df['TDP_W'] = df['Board Design__TDP'].str.extract('(\d+)').astype(float)

# D) Evolución de Memory Size
df['Memory_GB'] = df['Memory__Memory Size'].str.extract('(\d+)').astype(float)

# 3. VISUALIZACIONES INTERESANTES

# Ley de Moore en GPUs
plt.figure(figsize=(12, 6))
valid_data = df.dropna(subset=['Year', 'Transistors_M'])
plt.scatter(valid_data['Year'], valid_data['Transistors_M'], alpha=0.6, s=50)
plt.yscale('log')
plt.title('Evolución de Transistores en GPUs (1986-2026)')
plt.xlabel('Año')
plt.ylabel('Transistores (Millones, escala log)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Comparación por fabricante
top_brands = df['Brand'].value_counts().head(5).index
df_brands = df[df['Brand'].isin(top_brands)]

# TDP por marca a lo largo del tiempo
plt.figure(figsize=(14, 6))
for brand in top_brands:
    brand_data = df_brands[(df_brands['Brand'] == brand) & df_brands['Year'].notna() & df_brands['TDP_W'].notna()]
    if len(brand_data) > 0:
        plt.scatter(brand_data['Year'], brand_data['TDP_W'], label=brand, alpha=0.6, s=50)
plt.title('Evolución del TDP por Fabricante')
plt.xlabel('Año')
plt.ylabel('TDP (Watts)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Evolución del tamaño de proceso por año
plt.figure(figsize=(12, 6))
process_data = df.dropna(subset=['Year', 'Process_nm'])
plt.scatter(process_data['Year'], process_data['Process_nm'], alpha=0.5, s=50)
plt.yscale('log')
plt.title('Evolución del Tamaño de Proceso en GPUs')
plt.xlabel('Año')
plt.ylabel('Tamaño de Proceso (nm, escala log)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Evolución del tamaño de memoria por año
plt.figure(figsize=(12, 6))
memory_data = df.dropna(subset=['Year', 'Memory_GB'])
plt.scatter(memory_data['Year'], memory_data['Memory_GB'], alpha=0.5, s=50)
plt.yscale('log')
plt.title('Evolución del Tamaño de Memoria en GPUs')
plt.xlabel('Año')
plt.ylabel('Tamaño de Memoria (GB, escala log)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

input("Presiona Enter para salir...")
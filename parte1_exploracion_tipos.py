"""
================================================================================
LABORATORIO 1 - PARTE 1
EXPLORACIÓN DE DATOS Y CLASIFICACIÓN DE VARIABLES

Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la Computación
CC3074 – Minería de Datos
Semestre I – 2026

Esta parte incluye:
1. (3 puntos) Exploración rápida de datos - Resumen del conjunto de datos
2. (5 puntos) Clasificación del tipo de cada variable
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def print_section(title, char="="):
    """Imprime un título de sección con formato"""
    print(f"\n{char*80}")
    print(f"{title.center(80)}")
    print(f"{char*80}\n")


def load_data(filename):
    """Carga el dataset con el encoding apropiado"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"✓ Archivo cargado exitosamente con encoding: {encoding}")
            print(f"  Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
            return df
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    raise Exception("Error: No se pudo cargar el archivo.")


# ============================================================================
# CARGA DE DATOS
# ============================================================================

print_section("LABORATORIO 1 - PARTE 1: EXPLORACIÓN Y CLASIFICACIÓN")

df = load_data("movies_2026.csv")


# ============================================================================
# 1. EXPLORACIÓN RÁPIDA DE DATOS (3 puntos)
# ============================================================================

print_section("1. EXPLORACIÓN RÁPIDA DE DATOS", "-")

print("📊 RESUMEN DEL DATASET:\n")
summary = {
    "Filas (registros)": df.shape[0],
    "Columnas (variables)": df.shape[1],
    "Columnas con valores nulos": df.isnull().any().sum(),
    "Total de valores nulos": df.isnull().sum().sum(),
    "Registros duplicados": df.duplicated().sum(),
    "Memoria utilizada (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
}

for key, value in summary.items():
    print(f"  • {key}: {value:,}")

print("\n📋 PRIMERAS 5 FILAS DEL DATASET:")
print(df.head().to_string())

print("\n📊 ESTADÍSTICAS DESCRIPTIVAS (Variables Numéricas):")
print(df.describe().to_string())

print("\n🔍 VALORES NULOS POR COLUMNA:")
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if len(null_counts) > 0:
    for col, count in null_counts.items():
        pct = (count / len(df)) * 100
        print(f"  • {col}: {count:,} ({pct:.2f}%)")
else:
    print("  ✓ No hay valores nulos en el dataset")

print("\n📋 TIPOS DE DATOS POR COLUMNA:")
print(df.dtypes)

print("\n📊 INFORMACIÓN GENERAL DEL DATASET:")
print(df.info())


# ============================================================================
# 2. TIPO DE VARIABLES (5 puntos)
# ============================================================================

print_section("2. CLASIFICACIÓN DE VARIABLES", "-")

variable_types = {
    "Cualitativas Nominales": [
        "genres", "homePage", "productionCompany", "productionCompanyCountry",
        "productionCountry", "video", "director", "actors", "actorsCharacter",
        "originalTitle", "title", "originalLanguage"
    ],
    "Cualitativas Ordinales": [
        "voteAvg"
    ],
    "Cuantitativas Discretas": [
        "id", "voteCount", "genresAmount", "productionCoAmount",
        "productionCountriesAmount", "actorsAmount", "castWomenAmount",
        "castMenAmount", "releaseYear"
    ],
    "Cuantitativas Continuas": [
        "budget", "revenue", "runtime", "popularity", "actorsPopularity"
    ]
}

print("📌 CLASIFICACIÓN DE VARIABLES:\n")

for var_type, variables in variable_types.items():
    print(f"\n{var_type}:")
    existing_vars = [v for v in variables if v in df.columns]
    for i, var in enumerate(existing_vars, 1):
        print(f"  {i}. {var}")
    if len(existing_vars) == 0:
        print("  (Ninguna variable encontrada en el dataset)")

print("\n" + "="*80)
print("EXPLICACIÓN DETALLADA DE CADA TIPO DE VARIABLE")
print("="*80)

print("\n📍 1. CUALITATIVAS NOMINALES:")
print("  Características:")
print("    • Representan categorías o etiquetas sin orden inherente")
print("    • No se pueden ordenar de menor a mayor")
print("    • Solo se pueden contar frecuencias y calcular modas")
print("\n  Ejemplos del dataset:")
print("    • genres: Categorías de géneros de películas (Action, Drama, Comedy)")
print("    • originalLanguage: Idioma original (en, es, fr)")
print("    • director: Nombre del director")
print("    • productionCountry: País de producción")
print("    • title: Título de la película")

print("\n📍 2. CUALITATIVAS ORDINALES:")
print("  Características:")
print("    • Representan categorías con un orden natural")
print("    • Se pueden ordenar de menor a mayor")
print("    • Las diferencias entre categorías no son necesariamente iguales")
print("\n  Ejemplos del dataset:")
print("    • voteAvg: Calificación promedio (1.0 < 5.0 < 10.0)")
print("    • Aunque es numérica, representa niveles de satisfacción ordenados")

print("\n📍 3. CUANTITATIVAS DISCRETAS:")
print("  Características:")
print("    • Valores numéricos enteros (conteos)")
print("    • Representan cantidades que no se pueden dividir")
print("    • Resultado de contar elementos")
print("\n  Ejemplos del dataset:")
print("    • voteCount: Número de votos (1, 2, 3, ... no puede ser 2.5 votos)")
print("    • actorsAmount: Cantidad de actores (5, 10, 15 actores)")
print("    • releaseYear: Año de lanzamiento (2020, 2021, 2022)")
print("    • castWomenAmount: Número de mujeres en el reparto")

print("\n📍 4. CUANTITATIVAS CONTINUAS:")
print("  Características:")
print("    • Valores numéricos que pueden tener decimales")
print("    • Representan mediciones en una escala continua")
print("    • Resultado de medir (no de contar)")
print("\n  Ejemplos del dataset:")
print("    • budget: Presupuesto en dólares ($1,500,000.50)")
print("    • revenue: Ingresos ($2,345,678.90)")
print("    • runtime: Duración en minutos (120.5 minutos)")
print("    • popularity: Índice de popularidad (45.67)")
print("    • actorsPopularity: Popularidad promedio del elenco (12.34)")


print("\n" + "="*80)
print("RESUMEN DE CLASIFICACIÓN")
print("="*80)

total_vars = len(df.columns)
nominales = len([v for v in variable_types["Cualitativas Nominales"] if v in df.columns])
ordinales = len([v for v in variable_types["Cualitativas Ordinales"] if v in df.columns])
discretas = len([v for v in variable_types["Cuantitativas Discretas"] if v in df.columns])
continuas = len([v for v in variable_types["Cuantitativas Continuas"] if v in df.columns])

print(f"\nTotal de variables en el dataset: {total_vars}")
print(f"  • Cualitativas Nominales: {nominales} ({nominales/total_vars*100:.1f}%)")
print(f"  • Cualitativas Ordinales: {ordinales} ({ordinales/total_vars*100:.1f}%)")
print(f"  • Cuantitativas Discretas: {discretas} ({discretas/total_vars*100:.1f}%)")
print(f"  • Cuantitativas Continuas: {continuas} ({continuas/total_vars*100:.1f}%)")

print("\n💡 IMPLICACIONES PARA EL ANÁLISIS:")
print("  • Variables nominales: Solo podemos calcular frecuencias y moda")
print("  • Variables ordinales: Podemos usar mediana y percentiles")
print("  • Variables discretas: Media, mediana, moda y todas las estadísticas")
print("  • Variables continuas: Todo tipo de análisis estadístico avanzado")


# ============================================================================
# RESUMEN FINAL PARTE 1
# ============================================================================

print_section("RESUMEN FINAL - PARTE 1")

print("""
✅ PARTE 1 COMPLETADA

📊 Pregunta 1: Exploración rápida de datos
   ✓ Resumen estadístico generado
   ✓ Estructura del dataset analizada
   ✓ Valores nulos identificados
   ✓ Primeras filas visualizadas

📊 Pregunta 2: Clasificación de variables
   ✓ Variables nominales identificadas (12 variables)
   ✓ Variables ordinales identificadas (1 variable)
   ✓ Variables discretas identificadas (9 variables)
   ✓ Variables continuas identificadas (5 variables)
   ✓ Explicaciones detalladas proporcionadas

📁 PRÓXIMOS PASOS:
   → Ejecutar parte2_normalidad_frecuencias.py para análisis de normalidad
   → Ejecutar parte3_preguntas_4_1_a_4_9.py para preguntas específicas
   → Ejecutar parte4_preguntas_4_10_a_4_16.py para preguntas finales

""")

print("="*80)
print("PARTE 1 COMPLETADA".center(80))
print("="*80)

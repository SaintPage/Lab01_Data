"""
================================================================================
LABORATORIO 1 - PARTE 2
ANÁLISIS DE NORMALIDAD Y TABLAS DE FRECUENCIAS

Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la Computación
CC3074 – Minería de Datos
Semestre I – 2026

Esta parte incluye:
3. (6 puntos) Investigar si las variables cuantitativas siguen distribución normal
   y hacer tabla de frecuencias de las variables cualitativas. Explicar resultados.
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
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

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def print_section(title, char="="):
    """Imprime un título de sección con formato"""
    print(f"\n{char*80}")
    print(f"{title.center(80)}")
    print(f"{char*80}\n")


def save_figure(filename):
    """Guarda una figura con formato consistente"""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {filename}")


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

print_section("LABORATORIO 1 - PARTE 2: NORMALIDAD Y FRECUENCIAS")

df = load_data("movies_2026.csv")


# ============================================================================
# 3. ANÁLISIS DE NORMALIDAD Y FRECUENCIAS (6 puntos)
# ============================================================================

print_section("3. ANÁLISIS DE NORMALIDAD Y TABLAS DE FRECUENCIAS", "-")

# ============================================================================
# 3.A) PRUEBAS DE NORMALIDAD
# ============================================================================

print_section("3.A) PRUEBAS DE NORMALIDAD PARA VARIABLES CUANTITATIVAS", "·")

variables_cuantitativas = [
    'popularity', 'voteAvg', 'id', 'budget', 'revenue', 'runtime',
    'voteCount', 'genresAmount', 'productionCoAmount',
    'productionCountriesAmount', 'actorsAmount', 'castWomenAmount',
    'castMenAmount', 'releaseYear'
]

variables_cuantitativas = [var for var in variables_cuantitativas if var in df.columns]

print("📊 METODOLOGÍA:")
print("  1. Test de Shapiro-Wilk: Para muestras pequeñas (n ≤ 5000)")
print("     - Más preciso para muestras pequeñas")
print("     - Sensible a desviaciones de normalidad")
print("\n  2. Test de Kolmogorov-Smirnov: Para muestras grandes (n > 5000)")
print("     - Compara la distribución empírica con la normal")
print("     - Útil para grandes volúmenes de datos")

print("\n📋 HIPÓTESIS DE LAS PRUEBAS:")
print("  H₀ (Hipótesis Nula): Los datos siguen una distribución normal")
print("  H₁ (Hipótesis Alternativa): Los datos NO siguen una distribución normal")
print("\n  ⚠️  CRITERIO DE DECISIÓN:")
print("  Si p-value < 0.05 → Rechazamos H₀ → Los datos NO son normales")
print("  Si p-value ≥ 0.05 → No rechazamos H₀ → Los datos podrían ser normales")

resultados_normalidad = []

print("\n" + "="*80)
print("EJECUTANDO PRUEBAS DE NORMALIDAD...")
print("="*80)

for var in variables_cuantitativas:
    data = df[var].dropna()
    
    try:
        data = pd.to_numeric(data, errors='coerce')
        data = data[np.isfinite(data)]
    except:
        continue
    
    if len(data) > 3:
        # Test de Shapiro-Wilk para muestras pequeñas
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Test de Kolmogorov-Smirnov
        try:
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        except:
            ks_stat, ks_p = np.nan, np.nan
        
        # Determinar si es normal
        es_normal = False
        if not np.isnan(shapiro_p):
            es_normal = shapiro_p > 0.05
        elif not np.isnan(ks_p):
            es_normal = ks_p > 0.05
        
        resultados_normalidad.append({
            'Variable': var,
            'n': len(data),
            'Media': data.mean(),
            'Mediana': data.median(),
            'Desv.Est': data.std(),
            'Asimetría': data.skew(),
            'Curtosis': data.kurt(),
            'Shapiro_p': shapiro_p,
            'KS_p': ks_p,
            '¿Normal?': '✓ SÍ' if es_normal else '✗ NO'
        })

df_normalidad = pd.DataFrame(resultados_normalidad)

print("\n" + "="*80)
print("TABLA RESUMEN: RESULTADOS DE PRUEBAS DE NORMALIDAD")
print("="*80)
print(df_normalidad.to_string(index=False))

print("\n" + "="*80)
print("ANÁLISIS DETALLADO POR VARIABLE")
print("="*80)

for idx, row in df_normalidad.iterrows():
    print(f"\n{'─'*80}")
    print(f"📊 VARIABLE: {row['Variable'].upper()}")
    print(f"{'─'*80}")
    print(f"  📈 Estadísticas Descriptivas:")
    print(f"    • Tamaño de muestra: {row['n']:,}")
    print(f"    • Media: {row['Media']:.4f}")
    print(f"    • Mediana: {row['Mediana']:.4f}")
    print(f"    • Desviación estándar: {row['Desv.Est']:.4f}")
    print(f"    • Asimetría (Skewness): {row['Asimetría']:.4f}")
    print(f"    • Curtosis (Kurtosis): {row['Curtosis']:.4f}")
    
    # Interpretar asimetría
    print(f"\n  📐 Interpretación de Asimetría:")
    if abs(row['Asimetría']) < 0.5:
        print(f"    → Distribución aproximadamente simétrica")
    elif row['Asimetría'] > 0:
        print(f"    → Distribución sesgada a la DERECHA (cola larga hacia valores altos)")
        print(f"    → Mayoría de datos concentrados en valores bajos")
    else:
        print(f"    → Distribución sesgada a la IZQUIERDA (cola larga hacia valores bajos)")
        print(f"    → Mayoría de datos concentrados en valores altos")
    
    # Interpretar curtosis
    print(f"\n  📊 Interpretación de Curtosis:")
    if abs(row['Curtosis']) < 0.5:
        print(f"    → Distribución mesocúrtica (similar a la normal)")
    elif row['Curtosis'] > 0:
        print(f"    → Distribución leptocúrtica (más puntiaguda, con colas pesadas)")
        print(f"    → Presencia de valores extremos (outliers)")
    else:
        print(f"    → Distribución platicúrtica (más aplanada)")
    
    if not np.isnan(row['Shapiro_p']):
        print(f"\n  🔬 Test de Shapiro-Wilk:")
        print(f"    • p-value: {row['Shapiro_p']:.6f}")
        if row['Shapiro_p'] < 0.05:
            print(f"    • Conclusión: p < 0.05 → ✗ RECHAZAMOS H₀ → NO es normal")
        else:
            print(f"    • Conclusión: p ≥ 0.05 → ✓ No rechazamos H₀ → Podría ser normal")
    
    if not np.isnan(row['KS_p']):
        print(f"\n  🔬 Test de Kolmogorov-Smirnov:")
        print(f"    • p-value: {row['KS_p']:.6f}")
        if row['KS_p'] < 0.05:
            print(f"    • Conclusión: p < 0.05 → ✗ RECHAZAMOS H₀ → NO es normal")
        else:
            print(f"    • Conclusión: p ≥ 0.05 → ✓ No rechazamos H₀ → Podría ser normal")
    
    print(f"\n  🎯 VEREDICTO FINAL: {row['¿Normal?']}")

print("\n" + "="*80)
print("RESUMEN GENERAL DE NORMALIDAD")
print("="*80)

normales = df_normalidad[df_normalidad['¿Normal?'] == '✓ SÍ'].shape[0]
no_normales = df_normalidad[df_normalidad['¿Normal?'] == '✗ NO'].shape[0]
total = len(df_normalidad)

print(f"\n📊 Estadísticas Generales:")
print(f"  • Variables analizadas: {total}")
print(f"  • Variables con distribución normal: {normales} ({normales/total*100:.1f}%)")
print(f"  • Variables SIN distribución normal: {no_normales} ({no_normales/total*100:.1f}%)")

if normales > 0:
    print(f"\n✓ Variables NORMALES:")
    for var in df_normalidad[df_normalidad['¿Normal?'] == '✓ SÍ']['Variable']:
        print(f"    • {var}")

if no_normales > 0:
    print(f"\n✗ Variables NO NORMALES:")
    for var in df_normalidad[df_normalidad['¿Normal?'] == '✗ NO']['Variable']:
        print(f"    • {var}")

print("\n" + "="*80)
print("💡 INTERPRETACIÓN Y EXPLICACIÓN DE RESULTADOS")
print("="*80)

print("""
📋 ¿Por qué la mayoría de variables NO son normales?

1. 🎬 NATURALEZA DE LOS DATOS DE PELÍCULAS:
   • La industria cinematográfica es altamente desigual
   • Pocas películas blockbusters generan ingresos masivos
   • La mayoría de películas tienen presupuestos e ingresos bajos
   • Esto genera distribuciones asimétricas con sesgo positivo

2. 💰 DISTRIBUCIONES CON SESGO POSITIVO:
   • Budget (Presupuesto): Muchas películas independientes con bajo presupuesto,
     pocas superproducciones con presupuestos enormes
   • Revenue (Ingresos): Similar patrón - mayoría con ingresos bajos/moderados,
     pocas con ingresos estratosféricos
   • Popularity: Pocas películas extremadamente populares

3. 🎯 VARIABLES CON VALORES EXTREMOS (OUTLIERS):
   • VoteCount: Pocas películas tienen miles de votos
   • ActorsAmount: Mayoría con pocos actores, algunas con elencos masivos
   • Runtime: Mayoría 90-120 min, pero existen películas muy largas o muy cortas

4. 📊 VARIABLES DISCRETAS LIMITADAS:
   • GenresAmount: Limitado a pocos valores (1, 2, 3 géneros)
   • ProductionCoAmount: Similar restricción natural
   • Estas difícilmente pueden ser normales por su naturaleza discreta

5. 🌍 IMPLICACIONES PARA EL ANÁLISIS ESTADÍSTICO:
   ✓ USAR: Estadísticas robustas (mediana, cuartiles, rangos intercuartílicos)
   ✓ USAR: Pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis, Spearman)
   ✓ CONSIDERAR: Transformaciones logarítmicas para normalizar datos
   ✗ EVITAR: Asumir normalidad para pruebas paramétricas (t-test, ANOVA, etc.)
   ✗ EVITAR: Usar solo la media como medida de tendencia central

6. 🔄 TRANSFORMACIONES RECOMENDADAS:
   • Logaritmo: Para budget, revenue, popularity
   • Raíz cuadrada: Para conteos (voteCount, actorsAmount)
   • Box-Cox: Para normalización general
""")

# Visualización de normalidad
print("\n" + "="*80)
print("📊 GENERANDO VISUALIZACIONES DE NORMALIDAD...")
print("="*80)

n_vars = len(variables_cuantitativas)
fig, axes = plt.subplots(n_vars, 2, figsize=(16, 4*n_vars))
fig.suptitle('Análisis de Normalidad - Histogramas y Q-Q Plots', fontsize=16, y=0.995)

for idx, var in enumerate(variables_cuantitativas):
    if var in df.columns:
        data = df[var].dropna()
        
        try:
            data = pd.to_numeric(data, errors='coerce')
            data = data[np.isfinite(data)]
        except:
            continue
        
        if len(data) > 0:
            # Histograma
            axes[idx, 0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', 
                                linewidth=2, label=f'Media: {data.mean():.2f}')
            axes[idx, 0].axvline(data.median(), color='green', linestyle='--', 
                                linewidth=2, label=f'Mediana: {data.median():.2f}')
            axes[idx, 0].set_title(f'{var} - Histograma', fontsize=10, fontweight='bold')
            axes[idx, 0].set_xlabel('Valor')
            axes[idx, 0].set_ylabel('Frecuencia')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Q-Q Plot
            stats.probplot(data, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'{var} - Q-Q Plot', fontsize=10, fontweight='bold')
            axes[idx, 1].grid(True, alpha=0.3)

save_figure('imagenes/parte2_01_normalidad_histogramas_qqplots.png')
plt.close()

print("✓ Gráficos de normalidad guardados")
print("\n💡 INTERPRETACIÓN DE GRÁFICOS:")
print("  • Histograma: Muestra la distribución de frecuencias")
print("    - Normal: Forma de campana simétrica")
print("    - No normal: Asimetría, múltiples picos, colas largas")
print("  • Q-Q Plot: Compara cuantiles teóricos vs observados")
print("    - Normal: Puntos alineados en la línea diagonal")
print("    - No normal: Desviaciones de la línea, curvaturas")


# ============================================================================
# 3.B) TABLAS DE FRECUENCIAS
# ============================================================================

print_section("3.B) TABLAS DE FRECUENCIAS DE VARIABLES CUALITATIVAS", "·")

# Crear variable mainGenre si no existe
if 'mainGenre' not in df.columns:
    df["mainGenre"] = df["genres"].str.split("|").str[0]

variables_cualitativas = ['originalLanguage', 'video', 'releaseYear', 'genresAmount', 'mainGenre']

for var in variables_cualitativas:
    if var in df.columns:
        print(f"\n{'='*80}")
        print(f"📊 TABLA DE FRECUENCIAS: {var.upper()}")
        print(f"{'='*80}")
        
        # Calcular frecuencias
        freq_abs = df[var].value_counts().sort_values(ascending=False)
        freq_rel = df[var].value_counts(normalize=True).sort_values(ascending=False) * 100
        freq_acum = freq_abs.cumsum()
        freq_rel_acum = freq_rel.cumsum()
        
        # Crear tabla
        tabla = pd.DataFrame({
            'Categoría': freq_abs.index,
            'Frec.Absoluta': freq_abs.values,
            'Frec.Relativa(%)': freq_rel.values,
            'Frec.Acumulada': freq_acum.values,
            'Frec.Rel.Acum(%)': freq_rel_acum.values
        })
        
        print("\n📋 TOP 20 CATEGORÍAS MÁS FRECUENTES:")
        print(tabla.head(20).to_string(index=False))
        
        if len(tabla) > 20:
            print(f"\n  ... y {len(tabla) - 20} categorías adicionales")
        
        print(f"\n📈 ESTADÍSTICAS DE LA VARIABLE:")
        print(f"  • Total de categorías únicas: {len(freq_abs):,}")
        print(f"  • Total de registros válidos: {freq_abs.sum():,}")
        print(f"  • Valores nulos: {df[var].isnull().sum():,}")
        print(f"  • Categoría más frecuente: {freq_abs.index[0]} ({freq_rel.values[0]:.2f}%)")
        print(f"  • Categoría menos frecuente: {freq_abs.index[-1]} ({freq_rel.values[-1]:.2f}%)")
        
        # Explicación específica por variable
        print(f"\n💡 INTERPRETACIÓN:")
        if var == 'originalLanguage':
            print(f"  • El inglés domina la producción cinematográfica global")
            print(f"  • Refleja la hegemonía de Hollywood en la industria")
            print(f"  • Otras lenguas representan nichos de mercado específicos")
        elif var == 'video':
            print(f"  • Indica si la película fue lanzada directamente a video/streaming")
            print(f"  • La mayoría son lanzamientos teatrales (cines)")
        elif var == 'releaseYear':
            print(f"  • Muestra la distribución temporal del dataset")
            print(f"  • Permite identificar tendencias y evolución de la industria")
        elif var == 'genresAmount':
            print(f"  • Indica complejidad y diversidad del contenido")
            print(f"  • Películas con múltiples géneros buscan atraer más audiencia")
        elif var == 'mainGenre':
            print(f"  • Género predominante define la categorización principal")
            print(f"  • Útil para análisis de mercado y preferencias")

# Visualización de frecuencias
print("\n" + "="*80)
print("📊 GENERANDO VISUALIZACIONES DE FRECUENCIAS...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Tablas de Frecuencia - Variables Cualitativas', fontsize=16)

# Idiomas originales
if 'originalLanguage' in df.columns:
    top_langs = df['originalLanguage'].value_counts().head(15)
    axes[0, 0].barh(range(len(top_langs)), top_langs.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_langs)))
    axes[0, 0].set_yticklabels(top_langs.index)
    axes[0, 0].set_title('Top 15 Idiomas Originales', fontweight='bold')
    axes[0, 0].set_xlabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].invert_yaxis()

# Películas por año
if 'releaseYear' in df.columns:
    year_counts = df['releaseYear'].value_counts().sort_index().tail(20)
    axes[0, 1].bar(range(len(year_counts)), year_counts.values, color='coral')
    axes[0, 1].set_xticks(range(len(year_counts)))
    years = [int(y) if not np.isnan(y) else 'N/A' for y in year_counts.index]
    axes[0, 1].set_xticklabels(years, rotation=45, ha='right')
    axes[0, 1].set_title('Películas por Año (últimos 20)', fontweight='bold')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

# Cantidad de géneros
if 'genresAmount' in df.columns:
    genre_counts = df['genresAmount'].value_counts().sort_index()
    axes[0, 2].bar(genre_counts.index, genre_counts.values, color='green', alpha=0.7)
    axes[0, 2].set_title('Cantidad de Géneros por Película', fontweight='bold')
    axes[0, 2].set_xlabel('Número de Géneros')
    axes[0, 2].set_ylabel('Frecuencia')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

# Video vs Teatral
if 'video' in df.columns:
    video_counts = df['video'].value_counts()
    colors = ['lightblue', 'lightcoral']
    wedges, texts, autotexts = axes[1, 0].pie(video_counts.values, labels=video_counts.index, 
                                                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 0].set_title('Distribución: Video vs Teatral', fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

# Géneros principales
if 'mainGenre' in df.columns:
    main_genres = df['mainGenre'].value_counts().head(10)
    axes[1, 1].barh(range(len(main_genres)), main_genres.values, color='purple', alpha=0.7)
    axes[1, 1].set_yticks(range(len(main_genres)))
    axes[1, 1].set_yticklabels(main_genres.index)
    axes[1, 1].set_title('Top 10 Géneros Principales', fontweight='bold')
    axes[1, 1].set_xlabel('Frecuencia')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()

# Cantidad de actores
if 'actorsAmount' in df.columns:
    axes[1, 2].hist(df['actorsAmount'].dropna(), bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Distribución de Cantidad de Actores', fontweight='bold')
    axes[1, 2].set_xlabel('Cantidad de Actores')
    axes[1, 2].set_ylabel('Frecuencia')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

save_figure('imagenes/parte2_02_tablas_frecuencias.png')
plt.close()

print("✓ Gráficos de frecuencias guardados")


# ============================================================================
# RESUMEN FINAL PARTE 2
# ============================================================================

print_section("RESUMEN FINAL - PARTE 2")

print(f"""
✅ PARTE 2 COMPLETADA

📊 Pregunta 3: Análisis de Normalidad y Frecuencias

   A) PRUEBAS DE NORMALIDAD:
      ✓ {total} variables cuantitativas analizadas
      ✓ Tests de Shapiro-Wilk y Kolmogorov-Smirnov aplicados
      ✓ {normales} variables con distribución normal
      ✓ {no_normales} variables sin distribución normal
      ✓ Interpretaciones y explicaciones detalladas
      ✓ Visualizaciones: Histogramas y Q-Q Plots generados

   B) TABLAS DE FRECUENCIAS:
      ✓ 5 variables cualitativas analizadas
      ✓ Frecuencias absolutas y relativas calculadas
      ✓ Frecuencias acumuladas incluidas
      ✓ Interpretaciones específicas por variable
      ✓ Visualizaciones: Gráficos de barras y pastel generados

📁 ARCHIVOS GENERADOS (en carpeta imagenes/):
   • parte2_01_normalidad_histogramas_qqplots.png
   • parte2_02_tablas_frecuencias.png

🎯 HALLAZGOS PRINCIPALES:
   • La mayoría de variables NO siguen distribución normal
   • Sesgo positivo predominante en variables monetarias
   • Presencia de outliers en múltiples variables
   • Se recomienda usar estadísticas robustas y pruebas no paramétricas
   • Idioma inglés domina en frecuencias de idiomas
   • Géneros múltiples son comunes en películas modernas

📊 PRÓXIMOS PASOS:
   → Ejecutar parte3_preguntas_4_1_a_4_9.py para preguntas específicas
   → Ejecutar parte4_preguntas_4_10_a_4_16.py para preguntas finales
""")

print("="*80)
print("PARTE 2 COMPLETADA".center(80))
print("="*80)

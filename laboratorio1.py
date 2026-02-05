import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Cambiar al directorio del script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("movies_2026.csv", encoding="latin1")


##PREGUNTA 1
summary = {
    "filas": df.shape[0],
    "columnas": df.shape[1],
    "columnas_con_nulos": df.isnull().any().sum(),
    "total_nulos": df.isnull().sum().sum(),
    "duplicados": df.duplicated().sum()
}

print(summary)

##PREGUNTA 4.1
top10_budget = (
    df[["title", "budget"]]
    .sort_values(by="budget", ascending=False)
    .head(10)
)

print("Pregunta 4.1: Las 10 películas con mayor presupuesto")
print(top10_budget)


##Pregunta 4.2
top10_revenue = (
    df[["title", "revenue"]]
    .sort_values(by="revenue", ascending=False)
    .head(10)
)
print("Pregunta 4.2: Las 10 películas que más ingresos tuvieron")
print(top10_revenue)

##Pregunta 4.3
# Película con mayor número de votos
most_votes = (
    df[["title", "voteCount"]]
    .sort_values(by="voteCount", ascending=False)
    .head(1)
)

print("Pregunta 4.3: Película con mas votos")
print(most_votes)

##Pregunta 4.4
df_votes = df[df["voteCount"] > 0]

worst_movie = (
    df_votes[["title", "originalTitle", "voteAvg", "voteCount"]]
    .sort_values(by=["voteAvg", "voteCount"], ascending=[True, False])
    .head(1)
)
print("Pregunta 4.4: La peor película de acuerdo a los votos de todos los usuarios")
print(worst_movie)


##Pregunta 4.5
movies_per_year = df["releaseYear"].value_counts().sort_index()

most_movies_year = movies_per_year.idxmax()
most_movies_count = movies_per_year.max()

most_movies_year, most_movies_count

##grafico
plt.figure()
movies_per_year.plot(kind="bar")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Cantidad de películas")
plt.title("Cantidad de películas producidas por año")
plt.show()

##Pregunta 4.6
df["mainGenre"] = df["genres"].str.split("|").str[0]
df["releaseYear"] = pd.to_numeric(df["releaseYear"], errors="coerce")

recent_20 = df.sort_values(by="releaseYear", ascending=False).head(20)
genre_recent_20 = recent_20["mainGenre"].value_counts()

plt.figure()
genre_recent_20.plot(kind="bar")
plt.xlabel("Género principal")
plt.ylabel("Cantidad de películas")
plt.title("Género principal de las 20 películas más recientes")
plt.show()


genre_overall = df["mainGenre"].value_counts()
plt.figure()
genre_overall.head(10).plot(kind="bar")
plt.xlabel("Género principal")
plt.ylabel("Cantidad de películas")
plt.title("Géneros principales más frecuentes en el dataset")
plt.show()

longest_movies = df.sort_values(by="runtime", ascending=False).head(20)
genre_longest = longest_movies["mainGenre"].value_counts()

plt.figure()
genre_longest.plot(kind="bar")
plt.xlabel("Género principal")
plt.ylabel("Cantidad de películas")
plt.title("Género principal de las películas más largas")
plt.show()

##PRegunta 4.7
df["mainGenre"] = df["genres"].str.split("|").str[0]
df_revenue = df[df["revenue"] > 0]

avg_revenue_by_genre = (
    df_revenue
    .groupby("mainGenre")["revenue"]
    .mean()
    .sort_values(ascending=False) / 1_000_000
)

plt.figure()
avg_revenue_by_genre.head(10).plot(kind="bar")
plt.xlabel("Género principal")
plt.ylabel("Ingresos promedio (millones de USD)")
plt.title("Ingresos promedio por género principal")
plt.show()

##pregunta 4.8
df["actorsAmount"] = pd.to_numeric(df["actorsAmount"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

# Filtrar valores válidos
df_valid = df[(df["actorsAmount"] > 0) & (df["revenue"] > 0)].copy()

# Agrupar por cantidad de actores
df_valid["actorsGroup"] = pd.cut(
    df_valid["actorsAmount"],
    bins=[0, 5, 10, 20, 50, df_valid["actorsAmount"].max()],
    labels=["1–5", "6–10", "11–20", "21–50", "50+"]
)

# Ingresos en millones
df_valid["revenue_millions"] = df_valid["revenue"] / 1_000_000

# Boxplot
plt.figure()
df_valid.boxplot(column="revenue_millions", by="actorsGroup")
plt.xlabel("Cantidad de actores")
plt.ylabel("Ingresos (millones de USD)")
plt.title("Ingresos según cantidad de actores")
plt.suptitle("")
plt.show()


# -----------------------------
# 2) Evolución de actores en el tiempo
# -----------------------------
actors_by_year = (
    df.groupby("releaseYear")["actorsAmount"]
    .mean()
    .dropna()
)

plt.figure()
actors_by_year.plot(kind="line")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Cantidad promedio de actores")
plt.title("Evolución de la cantidad promedio de actores por año")
plt.show()

"""
Laboratorio 1 - Ejercicio 3
Analisis de Normalidad y Tablas de Frecuencia
Dataset: movies_2026.csv
"""



# Configuracion
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Cargar datos
print("="*80)
print("CARGANDO DATOS")
print("="*80)

encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv('movies_2026.csv', encoding=encoding)
        print(f"Archivo cargado con codificacion: {encoding}")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    print("Error: No se pudo cargar el archivo.")
    exit(1)

print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas\n")

# EJERCICIO 3: ANALISIS DE NORMALIDAD Y TABLAS DE FRECUENCIA

print("="*80)
print("EJERCICIO 3: ANALISIS DE NORMALIDAD Y TABLAS DE FRECUENCIA")
print("="*80)

# Variables cuantitativas
variables_cuantitativas = [
    'popularity', 'voteAvg', 'id', 'budget', 'revenue', 'runtime',
    'voteCount', 'genresAmount', 'productionCoAmount',
    'productionCountriesAmount', 'actorsAmount', 'castWomenAmount',
    'castMenAmount', 'releaseYear'
]

variables_cuantitativas = [var for var in variables_cuantitativas if var in df.columns]


# PARTE A: PRUEBAS DE NORMALIDAD

print("\n" + "="*80)
print("A. PRUEBAS DE NORMALIDAD PARA VARIABLES CUANTITATIVAS")
print("="*80)

print("\nPruebas estadisticas:")
print("  1. Shapiro-Wilk: Para muestras pequenas (n < 5000)")
print("  2. Kolmogorov-Smirnov: Para muestras grandes")
print("\nHipotesis:")
print("  H0: Los datos siguen una distribucion normal")
print("  H1: Los datos NO siguen una distribucion normal")
print("  Si p-value < 0.05, rechazamos H0\n")

resultados_normalidad = []

for var in variables_cuantitativas:
    if df[var].dtype in ['object', 'string']:
        continue
        
    data = df[var].dropna()
    
    try:
        data = pd.to_numeric(data, errors='coerce')
        data = data[np.isfinite(data)]
    except:
        continue
    
    if len(data) > 3:
        # Shapiro-Wilk (solo para muestras <= 5000)
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
            except:
                shapiro_stat, shapiro_p = np.nan, np.nan
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Kolmogorov-Smirnov
        try:
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        except:
            ks_stat, ks_p = np.nan, np.nan
        
        # Determinar normalidad
        es_normal = False
        if not np.isnan(shapiro_p):
            es_normal = shapiro_p > 0.05
        elif not np.isnan(ks_p):
            es_normal = ks_p > 0.05
        
        resultados_normalidad.append({
            'Variable': var,
            'n': len(data),
            'Media': data.mean(),
            'Desv.Est': data.std(),
            'Shapiro_p': shapiro_p,
            'KS_p': ks_p,
            'Normal': 'SI' if es_normal else 'NO'
        })

# Mostrar resultados
df_normalidad = pd.DataFrame(resultados_normalidad)
print("\n" + "="*80)
print("TABLA RESUMEN: PRUEBAS DE NORMALIDAD")
print("="*80)
print(df_normalidad.to_string(index=False))

# Explicacion detallada
print("\n" + "="*80)
print("EXPLICACION DETALLADA")
print("="*80)

for idx, row in df_normalidad.iterrows():
    print(f"\n{'-'*80}")
    print(f"VARIABLE: {row['Variable']}")
    print(f"{'-'*80}")
    print(f"  Tamano muestra: {row['n']:,}")
    print(f"  Media: {row['Media']:.4f}")
    print(f"  Desv.Est: {row['Desv.Est']:.4f}")
    
    if not np.isnan(row['Shapiro_p']):
        print(f"\n  Test Shapiro-Wilk:")
        print(f"    - p-value: {row['Shapiro_p']:.6f}")
        if row['Shapiro_p'] < 0.05:
            print(f"    - Interpretacion: p < 0.05 -> Rechazamos H0")
            print(f"    - Conclusion: NO es normal")
        else:
            print(f"    - Interpretacion: p >= 0.05 -> No rechazamos H0")
            print(f"    - Conclusion: Podria ser normal")
    
    if not np.isnan(row['KS_p']):
        print(f"\n  Test Kolmogorov-Smirnov:")
        print(f"    - p-value: {row['KS_p']:.6f}")
        if row['KS_p'] < 0.05:
            print(f"    - Interpretacion: p < 0.05 -> Rechazamos H0")
            print(f"    - Conclusion: NO es normal")
        else:
            print(f"    - Interpretacion: p >= 0.05 -> No rechazamos H0")
            print(f"    - Conclusion: Podria ser normal")
    
    print(f"\n  >>> VEREDICTO: {row['Normal']} es normal")

# Resumen
print("\n" + "="*80)
print("RESUMEN GENERAL")
print("="*80)
normales = df_normalidad[df_normalidad['Normal'] == 'SI'].shape[0]
no_normales = df_normalidad[df_normalidad['Normal'] == 'NO'].shape[0]
print(f"\nVariables analizadas: {len(df_normalidad)}")
print(f"Normales: {normales} ({normales/len(df_normalidad)*100:.1f}%)")
print(f"NO normales: {no_normales} ({no_normales/len(df_normalidad)*100:.1f}%)")

print("\nPor que ninguna variable es normal?")
print("  - Los datos de peliculas tienen distribuciones sesgadas")
print("  - Muchas peliculas con valores bajos (budget=0, revenue=0)")
print("  - Pocas peliculas con valores muy altos (blockbusters)")
print("  - Esto genera distribuciones asimetricas")

# PARTE B: TABLAS DE FRECUENCIA

print("\n" + "="*80)
print("B. TABLAS DE FRECUENCIA - VARIABLES CUALITATIVAS")
print("="*80)

variables_cualitativas = ['originalLanguage', 'video', 'releaseYear', 'genresAmount']

for var in variables_cualitativas:
    if var in df.columns:
        print(f"\n{'='*80}")
        print(f"TABLA: {var}")
        print(f"{'='*80}")
        
        # Calcular frecuencias
        freq_abs = df[var].value_counts().sort_values(ascending=False)
        freq_rel = df[var].value_counts(normalize=True).sort_values(ascending=False) * 100
        freq_acum = freq_abs.cumsum()
        freq_rel_acum = freq_rel.cumsum()
        
        # Crear tabla
        tabla = pd.DataFrame({
            'Categoria': freq_abs.index,
            'Frec.Abs': freq_abs.values,
            'Frec.Rel(%)': freq_rel.values,
            'Frec.Acum': freq_acum.values,
            'Frec.Rel.Acum(%)': freq_rel_acum.values
        })
        
        # Mostrar top 20
        print(tabla.head(20).to_string(index=False))
        
        if len(tabla) > 20:
            print(f"\n  ... y {len(tabla) - 20} categorias mas")
        
        print(f"\n  Estadisticas:")
        print(f"    - Categorias unicas: {len(freq_abs)}")
        print(f"    - Valores nulos: {df[var].isnull().sum()}")
        print(f"    - Mas frecuente: {freq_abs.index[0]} ({freq_rel.values[0]:.2f}%)")

# Interpretacion
print("\n" + "="*80)
print("INTERPRETACION DE TABLAS")
print("="*80)

print("\nQue significan las columnas?")
print("  - Frec.Abs: Cantidad de apariciones")
print("  - Frec.Rel: Porcentaje del total")
print("  - Frec.Acum: Suma acumulativa")
print("  - Frec.Rel.Acum: Porcentaje acumulativo")

print("\nHallazgos principales:")

if 'originalLanguage' in df.columns:
    top_lang = df['originalLanguage'].value_counts().head(3)
    print(f"\n  originalLanguage:")
    for lang, count in top_lang.items():
        pct = (count / len(df)) * 100
        print(f"    - {lang}: {count:,} peliculas ({pct:.1f}%)")

if 'video' in df.columns:
    video_false = (df['video'] == False).sum()
    video_true = (df['video'] == True).sum()
    print(f"\n  video:")
    print(f"    - FALSE: {video_false:,} ({video_false/len(df)*100:.1f}%)")
    print(f"    - TRUE: {video_true:,} ({video_true/len(df)*100:.1f}%)")

if 'releaseYear' in df.columns:
    top_years = df['releaseYear'].value_counts().head(3)
    print(f"\n  releaseYear:")
    for year, count in top_years.items():
        pct = (count / len(df)) * 100
        yr = int(year) if not np.isnan(year) else 'N/A'
        print(f"    - {yr}: {count:,} peliculas ({pct:.1f}%)")

if 'genresAmount' in df.columns:
    print(f"\n  genresAmount:")
    print(f"    - Media: {df['genresAmount'].mean():.2f} generos")
    print(f"    - Moda: {df['genresAmount'].mode()[0]} generos")
    print(f"    - Rango: {df['genresAmount'].min():.0f}-{df['genresAmount'].max():.0f}")

# VISUALIZACIONES


print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# 1. Histogramas y Q-Q plots
fig, axes = plt.subplots(len(variables_cuantitativas), 2, 
                         figsize=(16, 4*len(variables_cuantitativas)))
fig.suptitle('Analisis de Normalidad', fontsize=16, y=0.995)

for idx, var in enumerate(variables_cuantitativas):
    if var in df.columns and df[var].dtype in ['int64', 'float64']:
        data = df[var].dropna()
        
        try:
            data = pd.to_numeric(data, errors='coerce')
            data = data[np.isfinite(data)]
        except:
            continue
        
        if len(data) > 0:
            # Histograma
            axes[idx, 0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx, 0].set_title(f'{var} - Histograma')
            axes[idx, 0].set_xlabel('Valor')
            axes[idx, 0].set_ylabel('Frecuencia')
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', 
                                linewidth=2, label=f'Media: {data.mean():.2f}')
            axes[idx, 0].legend()
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'{var} - Q-Q Plot')
            axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ejercicio3_normalidad.png', dpi=300, bbox_inches='tight')
print("OK Grafico guardado: ejercicio3_normalidad.png")

# 2. Frecuencias
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Tablas de Frecuencia', fontsize=16)

# originalLanguage
if 'originalLanguage' in df.columns:
    top_langs = df['originalLanguage'].value_counts().head(15)
    axes[0, 0].barh(range(len(top_langs)), top_langs.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_langs)))
    axes[0, 0].set_yticklabels(top_langs.index)
    axes[0, 0].set_title('Top 15 Idiomas')
    axes[0, 0].set_xlabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].invert_yaxis()

# releaseYear
if 'releaseYear' in df.columns:
    year_counts = df['releaseYear'].value_counts().sort_index().tail(20)
    axes[0, 1].bar(range(len(year_counts)), year_counts.values, color='coral')
    axes[0, 1].set_xticks(range(len(year_counts)))
    years = [int(y) if not np.isnan(y) else 'N/A' for y in year_counts.index]
    axes[0, 1].set_xticklabels(years, rotation=45, ha='right')
    axes[0, 1].set_title('Peliculas por Anio (ultimos 20)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

# genresAmount
if 'genresAmount' in df.columns:
    genre_counts = df['genresAmount'].value_counts().sort_index()
    axes[1, 0].bar(genre_counts.index, genre_counts.values, color='green', alpha=0.7)
    axes[1, 0].set_title('Cantidad de Generos')
    axes[1, 0].set_xlabel('Numero de Generos')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

# video
if 'video' in df.columns:
    video_counts = df['video'].value_counts()
    colors = ['lightblue', 'lightcoral']
    axes[1, 1].pie(video_counts.values, labels=video_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Video vs Teatral')

plt.tight_layout()
plt.savefig('ejercicio3_frecuencias.png', dpi=300, bbox_inches='tight')
print("OK Grafico guardado: ejercicio3_frecuencias.png")

# CONCLUSIONES


print("CONCLUSIONES DEL EJERCICIO 3")

print("\n1. NORMALIDAD:")
print("   - Ninguna variable es normal (p < 0.05)")
print("   - Distribuciones sesgadas por outliers")
print("   - Tipico en datos de peliculas")

print("\n2. FRECUENCIAS:")
print("   - Ingles domina con ~60%")
print("   - 99.6% son peliculas teatrales")
print("   - Mayoria tiene 1-3 generos")
print("   - Concentracion en 2025-2026")

print("\n3. IMPLICACIONES:")
print("   - Usar pruebas no parametricas")
print("   - Considerar transformaciones")
print("   - Analizar outliers con cuidado")

print("EJERCICIO 3 COMPLETADO")
print("\nArchivos generados:")
print("  - ejercicio3_normalidad.png")
print("  - ejercicio3_frecuencias.png")
print("\nAnalisis finalizado!")



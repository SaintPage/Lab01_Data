"""
================================================================================
LABORATORIO 1 - PARTE 3
PREGUNTAS ESPECÍFICAS 4.1 - 4.9

Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la Computación
Minería de Datos
Semestre I – 2026

Esta parte incluye las preguntas:
4.1. (3 puntos) ¿Cuáles son las 10 películas que contaron con más presupuesto?
4.2. (3 puntos) ¿Cuáles son las 10 películas que más ingresos tuvieron?
4.3. (3 puntos) ¿Cuál es la película que más votos tuvo?
4.4. (3 puntos) ¿Cuál es la peor película de acuerdo a los votos de todos los usuarios?
4.5. (8 puntos) ¿Cuántas películas se hicieron en cada año? ¿En qué año se hicieron más películas?
4.6. (9 puntos) ¿Cuál es el género principal de las 20 películas más recientes?
4.7. (8 puntos) ¿Las películas de qué género principal obtuvieron mayores ganancias?
4.8. (3 puntos) ¿La cantidad de actores influye en los ingresos de las películas?
4.9. (3 puntos) ¿Es posible que la cantidad de hombres y mujeres en el reparto influya?
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# CONFIGURACIÓN INICIAL

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# FUNCIONES AUXILIARES

def print_section(title, char="="):
    """Imprime un título de sección con formato"""
    print(f"\n{char*80}")
    print(f"{title.center(80)}")
    print(f"{char*80}\n")


def save_figure(filename):
    """Guarda una figura con formato consistente"""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: {filename}")


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


# CARGA DE DATOS

print_section("LABORATORIO 1 - PARTE 3: PREGUNTAS 4.1 - 4.9")

df = load_data("movies_2026.csv")

# Crear variable mainGenre si no existe
if 'mainGenre' not in df.columns:
    df["mainGenre"] = df["genres"].str.split("|").str[0]


# 4.1. TOP 10 PELÍCULAS CON MAYOR PRESUPUESTO

print_section("4.1. ¿CUÁLES SON LAS 10 PELÍCULAS CON MÁS PRESUPUESTO?", "·")

top10_budget = df.nlargest(10, 'budget')[['title', 'budget', 'releaseYear', 'director', 'mainGenre']]
top10_budget['budget_millions'] = top10_budget['budget'] / 1_000_000

print("🎬 TOP 10 PELÍCULAS CON MAYOR PRESUPUESTO:\n")
for i, (idx, row) in enumerate(top10_budget.iterrows(), 1):
    print(f"  {i}. {row['title']}")
    print(f"      Presupuesto: ${row['budget_millions']:.2f} millones")
    print(f"      Director: {row['director']}")
    print(f"      Género: {row['mainGenre']}")
    print(f"      Año: {int(row['releaseYear']) if pd.notna(row['releaseYear']) else 'N/A'}\n")

print("📊 ANÁLISIS:")
total_top10 = top10_budget['budget_millions'].sum()
promedio_top10 = top10_budget['budget_millions'].mean()
print(f"   Presupuesto total (Top 10): ${total_top10:,.2f} millones")
print(f"   Presupuesto promedio (Top 10): ${promedio_top10:,.2f} millones")
print(f"   Presupuesto más alto: ${top10_budget['budget_millions'].max():,.2f} millones")
print(f"   Presupuesto más bajo (del Top 10): ${top10_budget['budget_millions'].min():,.2f} millones")

# Gráfico
plt.figure(figsize=(14, 8))
plt.barh(range(10), top10_budget['budget_millions'].values[::-1], color='gold', alpha=0.8)
plt.yticks(range(10), top10_budget['title'].values[::-1])
plt.xlabel('Presupuesto (Millones USD)', fontsize=12, fontweight='bold')
plt.title('Top 10 Películas con Mayor Presupuesto', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
save_figure('imagenes/parte3_01_top10_presupuesto.png')
plt.close()



# 4.2. TOP 10 PELÍCULAS CON MAYORES INGRESOS


print_section("4.2. ¿CUÁLES SON LAS 10 PELÍCULAS CON MAYORES INGRESOS?", "·")

top10_revenue = df.nlargest(10, 'revenue')[['title', 'revenue', 'budget', 'releaseYear', 'director', 'mainGenre']]
top10_revenue['revenue_millions'] = top10_revenue['revenue'] / 1_000_000
top10_revenue['budget_millions'] = top10_revenue['budget'] / 1_000_000
top10_revenue['profit_millions'] = top10_revenue['revenue_millions'] - top10_revenue['budget_millions']
top10_revenue['roi'] = (top10_revenue['profit_millions'] / top10_revenue['budget_millions'] * 100).fillna(0)

print("💰 TOP 10 PELÍCULAS CON MAYORES INGRESOS:\n")
for i, (idx, row) in enumerate(top10_revenue.iterrows(), 1):
    print(f"  {i}. {row['title']}")
    print(f"      Ingresos: ${row['revenue_millions']:.2f} millones")
    print(f"      Presupuesto: ${row['budget_millions']:.2f} millones")
    print(f"      Ganancia: ${row['profit_millions']:.2f} millones")
    print(f"      ROI: {row['roi']:.1f}%")
    print(f"      Director: {row['director']}")
    print(f"      Género: {row['mainGenre']}")
    print(f"      Año: {int(row['releaseYear']) if pd.notna(row['releaseYear']) else 'N/A'}\n")

print(" ANÁLISIS:")
total_ingresos = top10_revenue['revenue_millions'].sum()
promedio_ingresos = top10_revenue['revenue_millions'].mean()
print(f"  • Ingresos totales (Top 10): ${total_ingresos:,.2f} millones")
print(f"  • Ingresos promedio (Top 10): ${promedio_ingresos:,.2f} millones")
print(f"  • Ganancia total (Top 10): ${top10_revenue['profit_millions'].sum():,.2f} millones")
print(f"  • ROI promedio (Top 10): {top10_revenue['roi'].mean():.1f}%")

# Gráfico
plt.figure(figsize=(14, 8))
plt.barh(range(10), top10_revenue['revenue_millions'].values[::-1], color='green', alpha=0.8)
plt.yticks(range(10), top10_revenue['title'].values[::-1])
plt.xlabel('Ingresos (Millones USD)', fontsize=12, fontweight='bold')
plt.title('Top 10 Películas con Mayores Ingresos', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
save_figure('imagenes/parte3_02_top10_ingresos.png')
plt.close()


# 4.3. PELÍCULA CON MÁS VOTOS

print_section("4.3. ¿CUÁL ES LA PELÍCULA QUE MÁS VOTOS TUVO?", "·")

most_voted = df.nlargest(1, 'voteCount')[['title', 'voteCount', 'voteAvg', 'releaseYear', 
                                            'director', 'mainGenre', 'revenue']]

print(" PELÍCULA CON MÁS VOTOS:\n")
for idx, row in most_voted.iterrows():
    print(f"   Título: {row['title']}")
    print(f"    Votos: {int(row['voteCount']):,}")
    print(f"   Calificación promedio: {row['voteAvg']:.2f}/10")
    print(f"   Director: {row['director']}")
    print(f"   Género: {row['mainGenre']}")
    print(f"   Año: {int(row['releaseYear']) if pd.notna(row['releaseYear']) else 'N/A'}")
    if row['revenue'] > 0:
        print(f"   Ingresos: ${row['revenue']/1_000_000:.2f} millones")

print("\n INTERPRETACIÓN:")
print("   Alta cantidad de votos indica gran popularidad y audiencia masiva")
print("   Los usuarios se sintieron motivados a calificar la película")
print("   Refleja el impacto cultural y alcance de la película")

# Top 10 más votadas
top10_votes = df.nlargest(10, 'voteCount')[['title', 'voteCount', 'voteAvg']]
plt.figure(figsize=(14, 8))
plt.barh(range(10), top10_votes['voteCount'].values[::-1], color='coral', alpha=0.8)
plt.yticks(range(10), top10_votes['title'].values[::-1])
plt.xlabel('Cantidad de Votos', fontsize=12, fontweight='bold')
plt.title('Top 10 Películas con Más Votos', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
save_figure('imagenes/parte3_03_mas_votadas.png')
plt.close()


# 4.4. PEOR PELÍCULA SEGÚN VOTOS

print_section("4.4. ¿CUÁL ES LA PEOR PELÍCULA SEGÚN LOS VOTOS?", "·")

# Filtrar solo películas con al menos 50 votos para evitar sesgos
df_votes = df[df["voteCount"] >= 50]
worst_movie = df_votes.nsmallest(1, 'voteAvg')[['title', 'originalTitle', 'voteAvg', 'voteCount', 
                                                  'releaseYear', 'director', 'mainGenre']]

print(" PEOR PELÍCULA (con al menos 50 votos para validez estadística):\n")
for idx, row in worst_movie.iterrows():
    print(f"   Título: {row['title']}")
    print(f"   Título original: {row['originalTitle']}")
    print(f"   Calificación: {row['voteAvg']:.2f}/10")
    print(f"    Votos: {int(row['voteCount']):,}")
    print(f"   Director: {row['director']}")
    print(f"   Género: {row['mainGenre']}")
    print(f"   Año: {int(row['releaseYear']) if pd.notna(row['releaseYear']) else 'N/A'}")

print("\n INTERPRETACIÓN:")
print("   Se requieren al menos 50 votos para evitar outliers estadísticos")
print("   Una película con pocos votos y baja calificación no es representativa")
print("   Calificación muy baja indica rechazo generalizado de la audiencia")

# Top 10 peores calificadas
worst10 = df_votes.nsmallest(10, 'voteAvg')[['title', 'voteAvg', 'voteCount']]
plt.figure(figsize=(14, 8))
plt.barh(range(10), worst10['voteAvg'].values[::-1], color='red', alpha=0.7)
plt.yticks(range(10), worst10['title'].values[::-1])
plt.xlabel('Calificación Promedio', fontsize=12, fontweight='bold')
plt.title('Top 10 Películas Peor Calificadas (mín. 50 votos)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.xlim(0, 10)
save_figure('imagenes/parte3_04_peores_calificadas.png')
plt.close()


# 4.5. PELÍCULAS POR AÑO

print_section("4.5. ¿CUÁNTAS PELÍCULAS SE HICIERON EN CADA AÑO?", "·")

movies_per_year = df['releaseYear'].value_counts().sort_index()
most_movies_year = movies_per_year.idxmax()
most_movies_count = movies_per_year.max()

print(f" ESTADÍSTICAS POR AÑO:\n")
print(f"   Año con MÁS películas: {int(most_movies_year)}")
print(f"   Cantidad: {most_movies_count:,} películas")
print(f"\n   Año con MENOS películas: {int(movies_per_year.idxmin())}")
print(f"   Cantidad: {movies_per_year.min():,} películas")
print(f"\n   Promedio películas por año: {movies_per_year.mean():.1f}")
print(f"   Mediana películas por año: {movies_per_year.median():.1f}")
print(f"   Total de años en el dataset: {len(movies_per_year)}")

print(f"\n PELÍCULAS POR AÑO (Últimos 20 años):")
for year, count in movies_per_year.tail(20).items():
    print(f"  {int(year)}: {count:,} películas")

print("\n INTERPRETACIÓN:")
print("   El año con más películas refleja un boom en la industria")
print("   Puede correlacionarse con avances tecnológicos o eventos globales")
print("   Tendencia creciente indica expansión de la industria cinematográfica")

# Gráfico completo
plt.figure(figsize=(16, 6))
movies_per_year.plot(kind='bar', color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Año de Lanzamiento', fontsize=12, fontweight='bold')
plt.ylabel('Cantidad de Películas', fontsize=12, fontweight='bold')
plt.title('Cantidad de Películas Producidas por Año', fontsize=14, fontweight='bold')
plt.axhline(movies_per_year.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Promedio: {movies_per_year.mean():.0f}')
plt.axhline(movies_per_year.median(), color='green', linestyle='--', 
            linewidth=2, label=f'Mediana: {movies_per_year.median():.0f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save_figure('imagenes/parte3_05_peliculas_por_anio.png')
plt.close()


# 4.6. ANÁLISIS DE GÉNEROS

print_section("4.6. ANÁLISIS DE GÉNEROS PRINCIPALES", "·")

# 20 películas más recientes
recent_20 = df.nlargest(20, 'releaseYear')
genre_recent_20 = recent_20['mainGenre'].value_counts()

print(" GÉNERO DE LAS 20 PELÍCULAS MÁS RECIENTES:\n")
for genre, count in genre_recent_20.items():
    pct = (count / 20) * 100
    print(f"  • {genre}: {count} películas ({pct:.1f}%)")

# Género predominante en todo el dataset
genre_overall = df['mainGenre'].value_counts()
print(f"\n GÉNERO PREDOMINANTE EN TODO EL DATASET:")
print(f"   {genre_overall.index[0]}: {genre_overall.values[0]:,} películas ({genre_overall.values[0]/len(df)*100:.1f}%)")

print(f"\n TOP 10 GÉNEROS MÁS FRECUENTES:")
for i, (genre, count) in enumerate(genre_overall.head(10).items(), 1):
    pct = (count / len(df)) * 100
    print(f"  {i}. {genre}: {count:,} películas ({pct:.1f}%)")

# Películas más largas
longest_movies = df.nlargest(20, 'runtime')
genre_longest = longest_movies['mainGenre'].value_counts()

print(f"\n GÉNERO DE LAS 20 PELÍCULAS MÁS LARGAS:\n")
for genre, count in genre_longest.items():
    pct = (count / 20) * 100
    print(f"   {genre}: {count} películas ({pct:.1f}%)")

# Duración promedio por género
runtime_by_genre = df.groupby('mainGenre')['runtime'].mean().sort_values(ascending=False)
print(f"\n DURACIÓN PROMEDIO POR GÉNERO (Top 10):")
for genre, duration in runtime_by_genre.head(10).items():
    print(f"  • {genre}: {duration:.1f} minutos")

print("\n INTERPRETACIÓN:")
print("   Género de películas recientes muestra tendencias actuales del mercado")
print("   Género predominante refleja preferencias históricas de la industria")
print("   Géneros como Drama tienden a tener duraciones más largas")
print("   Action y Adventure suelen tener duraciones estándar (90-120 min)")

# Gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Género de las 20 más recientes
genre_recent_20.plot(kind='bar', ax=axes[0], color='coral', alpha=0.7)
axes[0].set_title('Género de las 20 Más Recientes', fontweight='bold')
axes[0].set_xlabel('Género')
axes[0].set_ylabel('Frecuencia')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Top 10 géneros más frecuentes
genre_overall.head(10).plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('Top 10 Géneros Más Frecuentes', fontweight='bold')
axes[1].set_xlabel('Género')
axes[1].set_ylabel('Cantidad de Películas')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

# Género de las 20 más largas
genre_longest.plot(kind='bar', ax=axes[2], color='purple', alpha=0.7)
axes[2].set_title('Género de las 20 Más Largas', fontweight='bold')
axes[2].set_xlabel('Género')
axes[2].set_ylabel('Frecuencia')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

save_figure('imagenes/parte3_06_analisis_generos.png')
plt.close()


# 4.7. INGRESOS POR GÉNERO

print_section("4.7. ¿QUÉ GÉNERO OBTUVO MAYORES GANANCIAS?", "·")

df_revenue = df[df['revenue'] > 0].copy()
df_revenue['revenue_millions'] = df_revenue['revenue'] / 1_000_000
df_revenue['budget_millions'] = df_revenue['budget'] / 1_000_000
df_revenue['profit_millions'] = df_revenue['revenue_millions'] - df_revenue['budget_millions']

# Ingresos promedio por género
avg_revenue_by_genre = df_revenue.groupby('mainGenre')['revenue_millions'].mean().sort_values(ascending=False)

# Ganancias totales por género
total_revenue_by_genre = df_revenue.groupby('mainGenre')['revenue_millions'].sum().sort_values(ascending=False)

# Ganancias netas por género
avg_profit_by_genre = df_revenue.groupby('mainGenre')['profit_millions'].mean().sort_values(ascending=False)

print("INGRESOS PROMEDIO POR GÉNERO (Top 10):\n")
for i, (genre, revenue) in enumerate(avg_revenue_by_genre.head(10).items(), 1):
    count = df_revenue[df_revenue['mainGenre'] == genre].shape[0]
    print(f"  {i}. {genre}: ${revenue:.2f} millones (basado en {count:,} películas)")

print("\n💸 INGRESOS TOTALES POR GÉNERO (Top 10):\n")
for i, (genre, revenue) in enumerate(total_revenue_by_genre.head(10).items(), 1):
    count = df_revenue[df_revenue['mainGenre'] == genre].shape[0]
    print(f"  {i}. {genre}: ${revenue:,.0f} millones totales ({count:,} películas)")

print("\n GANANCIA NETA PROMEDIO POR GÉNERO (Top 10):\n")
for i, (genre, profit) in enumerate(avg_profit_by_genre.head(10).items(), 1):
    print(f"  {i}. {genre}: ${profit:.2f} millones de ganancia promedio")

print("\n INTERPRETACIÓN:")
print("   Géneros con altos ingresos promedio pero pocas películas son nichos rentables")
print("   Géneros con altos ingresos totales dominan el mercado")
print("   Ganancia neta muestra eficiencia: ingresos menos presupuesto")
print("   Action y Adventure suelen tener alto ROI por su atractivo masivo")

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Ingresos promedio
avg_revenue_by_genre.head(10).plot(kind='bar', ax=axes[0], color='teal', alpha=0.7)
axes[0].set_title('Ingresos Promedio por Género (Top 10)', fontweight='bold')
axes[0].set_xlabel('Género Principal')
axes[0].set_ylabel('Ingresos Promedio (millones USD)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Ingresos totales
total_revenue_by_genre.head(10).plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('Ingresos Totales por Género (Top 10)', fontweight='bold')
axes[1].set_xlabel('Género Principal')
axes[1].set_ylabel('Ingresos Totales (millones USD)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

save_figure('imagenes/parte3_07_ingresos_por_genero.png')
plt.close()


# 4.8. CANTIDAD DE ACTORES VS INGRESOS

print_section("4.8. ¿LA CANTIDAD DE ACTORES INFLUYE EN LOS INGRESOS?", "·")

df_actors = df[(df['actorsAmount'] > 0) & (df['revenue'] > 0)].copy()
df_actors['revenue_millions'] = df_actors['revenue'] / 1_000_000

# Correlación
corr_actors = df_actors['actorsAmount'].corr(df_actors['revenue_millions'])
print(f" CORRELACIÓN actoresAmount vs ingresos: {corr_actors:.4f}\n")

if abs(corr_actors) < 0.3:
    interpretacion_corr = "débil o casi nula"
elif abs(corr_actors) < 0.5:
    interpretacion_corr = "moderada"
elif abs(corr_actors) < 0.7:
    interpretacion_corr = "considerable"
else:
    interpretacion_corr = "fuerte"

print(f"  • La correlación es {interpretacion_corr}")
if corr_actors > 0:
    print(f"  • Correlación positiva: más actores tiende a asociarse con mayores ingresos")
else:
    print(f"  • Correlación negativa: más actores tiende a asociarse con menores ingresos")

# Agrupar por rangos de actores
df_actors['actorsGroup'] = pd.cut(df_actors['actorsAmount'],
                                    bins=[0, 5, 10, 20, 50, df_actors['actorsAmount'].max()],
                                    labels=['1-5', '6-10', '11-20', '21-50', '50+'])

actors_group_stats = df_actors.groupby('actorsGroup')['revenue_millions'].agg(['mean', 'median', 'count'])

print(f"\n INGRESOS POR RANGO DE ACTORES:\n")
for group, row in actors_group_stats.iterrows():
    print(f"  {group} actores:")
    print(f"     Promedio: ${row['mean']:.2f} millones")
    print(f"     Mediana: ${row['median']:.2f} millones")
    print(f"     Películas: {int(row['count']):,}")

# Evolución de cantidad de actores por año
actors_by_year = df.groupby('releaseYear')['actorsAmount'].mean()
recent_avg = actors_by_year.tail(10).mean()
old_avg = actors_by_year.head(10).mean()
trend_pct = ((recent_avg / old_avg) - 1) * 100

print(f"\n EVOLUCIÓN TEMPORAL:")
print(f"   Promedio actores (primeros 10 años): {old_avg:.1f}")
print(f"   Promedio actores (últimos 10 años): {recent_avg:.1f}")
print(f"   Cambio: {trend_pct:+.1f}%")

if trend_pct > 0:
    print(f"   SÍ, se han hecho películas con más actores en años recientes")
else:
    print(f"   NO, la cantidad de actores se ha mantenido o reducido")

print("\n INTERPRETACIÓN:")
print("   Elencos grandes pueden indicar películas de alto presupuesto")
print("   No necesariamente garantizan éxito comercial")
print("   La calidad del guión y dirección son factores más determinantes")

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot ingresos por rango de actores
df_actors.boxplot(column='revenue_millions', by='actorsGroup', ax=axes[0])
axes[0].set_title('Ingresos vs Cantidad de Actores', fontweight='bold')
axes[0].set_xlabel('Rango de Actores')
axes[0].set_ylabel('Ingresos (millones USD)')
axes[0].grid(True, alpha=0.3)
plt.suptitle('')

# Evolución temporal
actors_by_year.plot(kind='line', ax=axes[1], marker='o', color='blue', linewidth=2)
axes[1].set_title('Evolución de Cantidad Promedio de Actores', fontweight='bold')
axes[1].set_xlabel('Año')
axes[1].set_ylabel('Cantidad Promedio de Actores')
axes[1].grid(True, alpha=0.3)

save_figure('imagenes/parte3_08_actores_vs_ingresos.png')
plt.close()


# 4.9. GÉNERO DEL REPARTO VS INGRESOS Y POPULARIDAD

print_section("4.9. ¿INFLUYE EL GÉNERO DEL REPARTO EN POPULARIDAD E INGRESOS?", "·")

df_cast = df[(df['castWomenAmount'].notna()) & (df['castMenAmount'].notna()) & (df['revenue'] > 0)].copy()
df_cast['revenue_millions'] = df_cast['revenue'] / 1_000_000
df_cast['total_cast'] = df_cast['castWomenAmount'] + df_cast['castMenAmount']
df_cast['women_pct'] = (df_cast['castWomenAmount'] / df_cast['total_cast'] * 100).fillna(0)
df_cast = df_cast[df_cast['total_cast'] > 0]

# Correlaciones
corr_women_revenue = df_cast['castWomenAmount'].corr(df_cast['revenue_millions'])
corr_men_revenue = df_cast['castMenAmount'].corr(df_cast['revenue_millions'])
corr_women_pop = df_cast['castWomenAmount'].corr(df_cast['popularity'])
corr_men_pop = df_cast['castMenAmount'].corr(df_cast['popularity'])

print(f" CORRELACIONES CON INGRESOS:\n")
print(f"   Cantidad de mujeres vs ingresos: {corr_women_revenue:.4f}")
print(f"   Cantidad de hombres vs ingresos: {corr_men_revenue:.4f}")

print(f"\n CORRELACIONES CON POPULARIDAD:\n")
print(f"   Cantidad de mujeres vs popularidad: {corr_women_pop:.4f}")
print(f"   Cantidad de hombres vs popularidad: {corr_men_pop:.4f}")

# Estadísticas generales
print(f"\n ESTADÍSTICAS GENERALES DEL REPARTO:\n")
print(f"   Promedio mujeres por película: {df_cast['castWomenAmount'].mean():.1f}")
print(f"   Promedio hombres por película: {df_cast['castMenAmount'].mean():.1f}")
print(f"   Porcentaje promedio de mujeres: {df_cast['women_pct'].mean():.1f}%")
print(f"   Mediana mujeres: {df_cast['castWomenAmount'].median():.0f}")
print(f"   Mediana hombres: {df_cast['castMenAmount'].median():.0f}")

# Categorizar por porcentaje de mujeres
df_cast['women_category'] = pd.cut(df_cast['women_pct'],
                                     bins=[0, 25, 50, 75, 100],
                                     labels=['0-25%', '25-50%', '50-75%', '75-100%'])

revenue_by_women = df_cast.groupby('women_category')['revenue_millions'].agg(['mean', 'median', 'count'])
popularity_by_women = df_cast.groupby('women_category')['popularity'].agg(['mean', 'median'])

print(f"\n INGRESOS POR % DE MUJERES EN EL REPARTO:\n")
for cat, row in revenue_by_women.iterrows():
    print(f"  {cat} mujeres:")
    print(f"     Ingresos promedio: ${row['mean']:.2f} millones")
    print(f"     Ingresos mediana: ${row['median']:.2f} millones")
    print(f"     Películas: {int(row['count']):,}")

print(f"\n POPULARIDAD POR % DE MUJERES EN EL REPARTO:\n")
for cat, row in popularity_by_women.iterrows():
    print(f"  {cat} mujeres:")
    print(f"     Popularidad promedio: {row['mean']:.2f}")
    print(f"     Popularidad mediana: {row['median']:.2f}")

print("\n INTERPRETACIÓN:")
if abs(corr_women_revenue) < 0.2 and abs(corr_men_revenue) < 0.2:
    print("   NO hay una correlación significativa entre género del reparto e ingresos")
    print("   La composición de género del elenco NO es un factor determinante del éxito")
else:
    print("   SÍ existe cierta correlación entre género del reparto e ingresos")
    if corr_women_revenue > corr_men_revenue:
        print("   Mayor presencia de mujeres se asocia ligeramente con mejores ingresos")
    else:
        print("   Mayor presencia de hombres se asocia ligeramente con mejores ingresos")

print("\n   Factores más importantes para el éxito:")
print("     Calidad del guión y dirección")
print("     Popularidad individual de los actores")
print("     Presupuesto de marketing")
print("     Género de la película")
print("     Época de lanzamiento")

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Promedio hombres vs mujeres
means = [df_cast['castWomenAmount'].mean(), df_cast['castMenAmount'].mean()]
axes[0, 0].bar(['Mujeres', 'Hombres'], means, color=['pink', 'lightblue'], alpha=0.7)
axes[0, 0].set_title('Promedio de Actores por Género', fontweight='bold')
axes[0, 0].set_ylabel('Cantidad Promedio')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Ingresos por % de mujeres
revenue_by_women['mean'].plot(kind='bar', ax=axes[0, 1], color='purple', alpha=0.7)
axes[0, 1].set_title('Ingresos Promedio por % de Mujeres', fontweight='bold')
axes[0, 1].set_xlabel('% de Mujeres en Reparto')
axes[0, 1].set_ylabel('Ingresos (millones USD)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Popularidad por % de mujeres
popularity_by_women['mean'].plot(kind='bar', ax=axes[1, 0], color='orange', alpha=0.7)
axes[1, 0].set_title('Popularidad Promedio por % de Mujeres', fontweight='bold')
axes[1, 0].set_xlabel('% de Mujeres en Reparto')
axes[1, 0].set_ylabel('Popularidad')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Scatter plot mujeres vs hombres coloreado por ingresos
scatter = axes[1, 1].scatter(df_cast['castMenAmount'], df_cast['castWomenAmount'], 
                             c=df_cast['revenue_millions'], cmap='viridis', alpha=0.5, s=50)
axes[1, 1].set_title('Distribución de Género en Reparto', fontweight='bold')
axes[1, 1].set_xlabel('Cantidad de Hombres')
axes[1, 1].set_ylabel('Cantidad de Mujeres')
plt.colorbar(scatter, ax=axes[1, 1], label='Ingresos (millones USD)')
axes[1, 1].grid(True, alpha=0.3)

save_figure('imagenes/parte3_09_genero_reparto_analisis.png')
plt.close()



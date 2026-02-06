"""
LABORATORIO 1 - PARTE 4
PREGUNTAS ESPECÍFICAS 4.10 - 4.16

Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la Computación
Minería de Datos
Semestre I – 2026

Esta parte incluye las preguntas:
4.10. (8 puntos) ¿Quiénes son los directores de las 20 películas mejor calificadas?
4.11. (8 puntos) ¿Cómo se correlacionan presupuestos con ingresos?
4.12. (5 puntos) ¿Se asocian ciertos meses de lanzamiento con mejores ingresos?
4.13. (6 puntos) ¿En qué meses se han visto lanzamientos con mejores ingresos?
4.14. (7 puntos) ¿Cómo se correlacionan las calificaciones con el éxito comercial?
4.15. (5 puntos) ¿Qué estrategias de marketing generan mejores resultados?
4.16. (4 puntos) ¿La popularidad del elenco está correlacionada con el éxito?
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
    print(f" Grafico guardado: {filename}")


def load_data(filename):
    """Carga el dataset con el encoding apropiado"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"[OK] Archivo cargado exitosamente con encoding: {encoding}")
            print(f"  Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
            return df
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    raise Exception("Error: No se pudo cargar el archivo.")


def parse_popularity(pop_str):
    """Convierte string de popularidad a promedio numérico"""
    if pd.isna(pop_str) or pop_str == "":
        return np.nan
    try:
        values = [float(x) for x in str(pop_str).split("|") if x.strip()]
        return np.mean(values) if values else np.nan
    except:
        return np.nan


print_section("LABORATORIO 1 - PARTE 4: PREGUNTAS 4.10 - 4.16")

df = load_data("movies_2026.csv")

# Crear variable mainGenre si no existe
if 'mainGenre' not in df.columns:
    df["mainGenre"] = df["genres"].str.split("|").str[0]


# 4.10. DIRECTORES DE LAS 20 MEJOR CALIFICADAS

print_section("4.10. DIRECTORES DE LAS 20 PELÍCULAS MEJOR CALIFICADAS", "·")

# Filtrar películas con mínimo de votos para validez estadística
df_rated = df[df['voteCount'] >= 100].copy()
top20_rated = df_rated.nlargest(20, 'voteAvg')[['title', 'director', 'voteAvg', 'voteCount', 
                                                  'releaseYear', 'mainGenre']]

print("⭐ TOP 20 PELÍCULAS MEJOR CALIFICADAS (mín. 100 votos):\n")
for i, (idx, row) in enumerate(top20_rated.iterrows(), 1):
    print(f"  {i}. {row['title']}")
    print(f"      Director: {row['director']}")
    print(f"      Calificación: {row['voteAvg']:.2f}/10")
    print(f"       Votos: {int(row['voteCount']):,}")
    print(f"       Género: {row['mainGenre'] if 'mainGenre' in row else 'N/A'}")
    print(f"        Año: {int(row['releaseYear']) if pd.notna(row['releaseYear']) else 'N/A'}\n")

# Análisis de directores
director_counts = top20_rated['director'].value_counts()
multi_directors = director_counts[director_counts > 1]

print(" ESTADÍSTICAS DE DIRECTORES:\n")
print(f"   Total de directores únicos: {len(director_counts)}")
print(f"   Directores con más de una película en el Top 20: {len(multi_directors)}")

if len(multi_directors) > 0:
    print(f"\n🏆 DIRECTORES CON MÁS DE UNA PELÍCULA EN EL TOP 20:\n")
    for director, count in multi_directors.items():
        movies = top20_rated[top20_rated['director'] == director]['title'].tolist()
        avg_rating = top20_rated[top20_rated['director'] == director]['voteAvg'].mean()
        print(f"   {director}: {count} películas (calificación promedio: {avg_rating:.2f})")
        for movie in movies:
            print(f"      - {movie}")
        print()

# Top directores en todo el dataset
top_directors_overall = df_rated.groupby('director').agg({
    'voteAvg': 'mean',
    'title': 'count'
}).rename(columns={'title': 'movies_count'})
top_directors_overall = top_directors_overall[top_directors_overall['movies_count'] >= 5]
top_directors_overall = top_directors_overall.sort_values('voteAvg', ascending=False).head(15)

print(" TOP 15 DIRECTORES CON MEJORES CALIFICACIONES PROMEDIO (mín. 5 películas):\n")
for director, row in top_directors_overall.iterrows():
    print(f"   {director}")
    print(f"      Calificación promedio: {row['voteAvg']:.2f}/10")
    print(f"      Número de películas: {int(row['movies_count'])}")

print("\n INTERPRETACIÓN:")
print("   Directores en el Top 20 representan la élite del cine")
print("   Consistencia en calidad indica maestría cinematográfica")
print("   Múltiples películas en Top 20 es extremadamente raro y valioso")

# Gráfico
plt.figure(figsize=(14, 8))
top20_rated_sorted = top20_rated.sort_values('voteAvg', ascending=True)
plt.barh(range(20), top20_rated_sorted['voteAvg'].values, color='gold', alpha=0.8)

# Crear etiquetas con título y director
labels = []
for _, row in top20_rated_sorted.iterrows():
    title = row['title']
    director = row['director']
    
    # Truncar título si es muy largo
    if len(title) > 30:
        title = title[:30] + "..."
    
    # Si el director tiene múltiples nombres (separados por comas o | ), tomar solo los primeros dos
    if pd.notna(director):
        if ',' in str(director):
            directors_list = str(director).split(',')[:2]
            director_short = ', '.join(directors_list)
        elif '|' in str(director):
            directors_list = str(director).split('|')[:2]
            director_short = ', '.join(directors_list)
        else:
            director_short = str(director)
        
        # Truncar director si es muy largo
        if len(director_short) > 25:
            director_short = director_short[:25] + "..."
        
        label = f"{title}\n({director_short})"
    else:
        label = title
    
    labels.append(label)

plt.yticks(range(20), labels, fontsize=8)
plt.xlabel('Calificación Promedio', fontsize=12, fontweight='bold')
plt.title('Top 20 Películas Mejor Calificadas', fontsize=14, fontweight='bold')
plt.xlim(7, 10)
plt.grid(True, alpha=0.3, axis='x')
save_figure('imagenes/parte4_01_top20_mejor_calificadas.png')
plt.close()

print_section("4.11. CORRELACIÓN ENTRE PRESUPUESTOS E INGRESOS", "·")

df_budget = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
df_budget['budget_millions'] = df_budget['budget'] / 1_000_000
df_budget['revenue_millions'] = df_budget['revenue'] / 1_000_000
df_budget['profit_millions'] = df_budget['revenue_millions'] - df_budget['budget_millions']
df_budget['roi'] = (df_budget['profit_millions'] / df_budget['budget_millions'] * 100)

# Correlación
correlation = df_budget['budget_millions'].corr(df_budget['revenue_millions'])

print(f" CORRELACIÓN PRESUPUESTO VS INGRESOS: {correlation:.4f}\n")

if correlation > 0.7:
    interpretacion = "FUERTE y POSITIVA"
    explicacion = "Existe una relación considerable: mayor presupuesto tiende a generar mayores ingresos"
elif correlation > 0.5:
    interpretacion = "MODERADA-FUERTE y POSITIVA"
    explicacion = "Existe relación notable: presupuestos altos frecuentemente generan altos ingresos"
elif correlation > 0.3:
    interpretacion = "MODERADA y POSITIVA"
    explicacion = "Hay cierta relación: presupuesto influye pero no es el único factor determinante"
else:
    interpretacion = "DÉBIL"
    explicacion = "Poca relación: el presupuesto no garantiza ingresos altos"

print(f"  • Interpretación: Correlación {interpretacion}")
print(f"  • {explicacion}")

# Categorizar por rangos de presupuesto
df_budget['budget_category'] = pd.cut(df_budget['budget_millions'],
                                        bins=[0, 10, 50, 100, 200, 500],
                                        labels=['Bajo (<10M)', 'Medio (10-50M)', 
                                               'Alto (50-100M)', 'Muy Alto (100-200M)', 
                                               'Blockbuster (>200M)'])

budget_stats = df_budget.groupby('budget_category').agg({
    'revenue_millions': ['mean', 'median', 'count'],
    'roi': 'mean'
})

print(f"\n INGRESOS POR RANGO DE PRESUPUESTO:\n")
for category in budget_stats.index:
    count = budget_stats.loc[category, ('revenue_millions', 'count')]
    avg_revenue = budget_stats.loc[category, ('revenue_millions', 'mean')]
    median_revenue = budget_stats.loc[category, ('revenue_millions', 'median')]
    avg_roi = budget_stats.loc[category, ('roi', 'mean')]
    
    print(f"  {category}:")
    print(f"     Ingresos promedio: ${avg_revenue:.2f}M")
    print(f"     Ingresos mediana: ${median_revenue:.2f}M")
    print(f"     ROI promedio: {avg_roi:.1f}%")
    print(f"     Películas: {int(count):,}\n")

print(" ¿ALTOS PRESUPUESTOS = ALTOS INGRESOS?")
if correlation > 0.5:
    print(f"   SÍ, generalmente:")
    print(f"     La correlación de {correlation:.2f} indica relación fuerte")
    print(f"     Presupuestos altos permiten:")
    print(f"       Efectos especiales de calidad")
    print(f"       Actores famosos")
    print(f"       Campañas de marketing masivas")
    print(f"       Locaciones y producción de alta calidad")
else:
    print(f"   NO necesariamente:")
    print(f"     La correlación de {correlation:.2f} indica relación moderada/débil")
    print(f"     Muchos factores adicionales influyen:")
    print(f"       Calidad del guión")
    print(f"       Momento del lanzamiento")
    print(f"       Competencia")
    print(f"       Recepción crítica")

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Diagrama de dispersión con línea de tendencia
axes[0, 0].scatter(df_budget['budget_millions'], df_budget['revenue_millions'], 
                   alpha=0.4, s=30, color='steelblue')
z = np.polyfit(df_budget['budget_millions'], df_budget['revenue_millions'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df_budget['budget_millions'], p(df_budget['budget_millions']), 
                "r--", linewidth=2, label=f'Línea de tendencia')
axes[0, 0].set_xlabel('Presupuesto (millones USD)', fontweight='bold')
axes[0, 0].set_ylabel('Ingresos (millones USD)', fontweight='bold')
axes[0, 0].set_title(f'Presupuesto vs Ingresos (r={correlation:.3f})', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histograma de presupuestos
axes[0, 1].hist(df_budget['budget_millions'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(df_budget['budget_millions'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: ${df_budget["budget_millions"].mean():.1f}M')
axes[0, 1].axvline(df_budget['budget_millions'].median(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mediana: ${df_budget["budget_millions"].median():.1f}M')
axes[0, 1].set_title('Distribución de Presupuestos', fontweight='bold')
axes[0, 1].set_xlabel('Presupuesto (millones USD)', fontweight='bold')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Histograma de ingresos
axes[1, 0].hist(df_budget['revenue_millions'], bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(df_budget['revenue_millions'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: ${df_budget["revenue_millions"].mean():.1f}M')
axes[1, 0].axvline(df_budget['revenue_millions'].median(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mediana: ${df_budget["revenue_millions"].median():.1f}M')
axes[1, 0].set_title('Distribución de Ingresos', fontweight='bold')
axes[1, 0].set_xlabel('Ingresos (millones USD)', fontweight='bold')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Boxplot de ingresos por categoría de presupuesto
df_budget.boxplot(column='revenue_millions', by='budget_category', ax=axes[1, 1])
axes[1, 1].set_title('Ingresos por Categoría de Presupuesto', fontweight='bold')
axes[1, 1].set_xlabel('Categoría de Presupuesto', fontweight='bold')
axes[1, 1].set_ylabel('Ingresos (millones USD)')
axes[1, 1].tick_params(axis='x', rotation=45)
plt.suptitle('')

save_figure('imagenes/parte4_02_presupuesto_vs_ingresos.png')
plt.close()


# 4.12 Y 4.13. MESES DE LANZAMIENTO E INGRESOS

print_section("4.12-4.13. MESES DE LANZAMIENTO E INGRESOS", "·")

df_release = df[df['revenue'] > 0].copy()
df_release['releaseDate'] = pd.to_datetime(df_release['releaseDate'], errors='coerce')
df_release['releaseMonth'] = df_release['releaseDate'].dt.month
df_release['revenue_millions'] = df_release['revenue'] / 1_000_000

meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
         7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}

# Ingresos promedio por mes
avg_revenue_by_month = df_release.groupby('releaseMonth')['revenue_millions'].mean().sort_values(ascending=False)
total_revenue_by_month = df_release.groupby('releaseMonth')['revenue_millions'].sum().sort_values(ascending=False)
movies_per_month = df_release.groupby('releaseMonth').size()
median_revenue_by_month = df_release.groupby('releaseMonth')['revenue_millions'].median()

print(" INGRESOS PROMEDIO POR MES (ordenado de mayor a menor):\n")
for i, (month, revenue) in enumerate(avg_revenue_by_month.items(), 1):
    count = movies_per_month[month]
    total = total_revenue_by_month[month]
    print(f"  {i}. {meses[int(month)]}: ${revenue:.2f}M promedio | ${total:,.0f}M total | {count:,} películas")

best_month_avg = avg_revenue_by_month.idxmax()
best_month_total = total_revenue_by_month.idxmax()

print(f"\n MEJORES MESES:")
print(f"   Mejor mes (ingresos promedio): {meses[int(best_month_avg)]} - ${avg_revenue_by_month[best_month_avg]:.2f}M")
print(f"   Mejor mes (ingresos totales): {meses[int(best_month_total)]} - ${total_revenue_by_month[best_month_total]:,.0f}M")

print(f"\n ESTADÍSTICAS DE LANZAMIENTOS:")
avg_movies = movies_per_month.mean()
print(f"   Promedio de películas por mes: {avg_movies:.2f}")
print(f"   Total de películas analizadas: {movies_per_month.sum():,}")

print(f"\n PELÍCULAS LANZADAS POR MES:\n")
for month in range(1, 13):
    count = movies_per_month.get(month, 0)
    pct = (count / movies_per_month.sum() * 100)
    print(f"   {meses[month]}: {count:,} películas ({pct:.1f}%)")

print("\n INTERPRETACIÓN:")
print("   Meses de verano y vacaciones (Mayo-Julio) suelen tener mejores ingresos:")
print("     Mayor audiencia disponible (vacaciones escolares)")
print("     Temporada de blockbusters")
print("   Noviembre-Diciembre son fuertes por:")
print("     Temporada de premios")
print("     Feriados y vacaciones")
print("   Enero-Febrero suelen ser más débiles:")
print("     Post-temporada navideña")
print("     Menor audiencia en cines")

# Identificar tendencias estacionales
verano = df_release[df_release['releaseMonth'].isin([6, 7, 8])]
invierno = df_release[df_release['releaseMonth'].isin([12, 1, 2])]
print(f"\ ANÁLISIS ESTACIONAL:")
print(f"   Verano (Jun-Ago): ${verano['revenue_millions'].mean():.2f}M promedio")
print(f"   Invierno (Dic-Feb): ${invierno['revenue_millions'].mean():.2f}M promedio")

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Ingresos promedio por mes
months_ordered = [avg_revenue_by_month.get(i, 0) for i in range(1, 13)]
axes[0, 0].bar(range(1, 13), months_ordered, color='teal', alpha=0.7)
axes[0, 0].set_xlabel('Mes', fontweight='bold')
axes[0, 0].set_ylabel('Ingresos Promedio (millones USD)')
axes[0, 0].set_title('Ingresos Promedio por Mes de Lanzamiento', fontweight='bold')
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels([meses[i][:3] for i in range(1, 13)])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Películas lanzadas por mes
axes[0, 1].bar(movies_per_month.index, movies_per_month.values, color='purple', alpha=0.7)
axes[0, 1].axhline(avg_movies, color='red', linestyle='--', linewidth=2, 
                   label=f'Promedio: {avg_movies:.1f}')
axes[0, 1].set_title('Películas Lanzadas por Mes', fontweight='bold')
axes[0, 1].set_xlabel('Mes', fontweight='bold')
axes[0, 1].set_ylabel('Cantidad de Películas')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels([meses[i][:3] for i in range(1, 13)])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Ingresos totales por mes
total_months_ordered = [total_revenue_by_month.get(i, 0) for i in range(1, 13)]
axes[1, 0].bar(range(1, 13), total_months_ordered, color='orange', alpha=0.7)
axes[1, 0].set_title('Ingresos Totales por Mes', fontweight='bold')
axes[1, 0].set_xlabel('Mes', fontweight='bold')
axes[1, 0].set_ylabel('Ingresos Totales (millones USD)')
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].set_xticklabels([meses[i][:3] for i in range(1, 13)])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Boxplot de ingresos por mes
df_release.boxplot(column='revenue_millions', by='releaseMonth', ax=axes[1, 1])
axes[1, 1].set_title('Distribución de Ingresos por Mes', fontweight='bold')
axes[1, 1].set_xlabel('Mes', fontweight='bold')
axes[1, 1].set_ylabel('Ingresos (millones USD)')
axes[1, 1].set_xticklabels([meses[i][:3] for i in range(1, 13)])
plt.suptitle('')

save_figure('imagenes/parte4_03_meses_lanzamiento_ingresos.png')
plt.close()


# 4.14. CALIFICACIONES VS ÉXITO COMERCIAL

print_section("4.14. CORRELACIÓN CALIFICACIONES VS ÉXITO COMERCIAL", "·")

df_ratings = df[(df['voteAvg'] > 0) & (df['revenue'] > 0) & (df['voteCount'] >= 50)].copy()
df_ratings['revenue_millions'] = df_ratings['revenue'] / 1_000_000

# Correlaciones
corr_rating_revenue = df_ratings['voteAvg'].corr(df_ratings['revenue_millions'])
corr_votes_revenue = df_ratings['voteCount'].corr(df_ratings['revenue_millions'])

print(f"    CORRELACIONES CON ÉXITO COMERCIAL:\n")
print(f"   Calificación (voteAvg) vs ingresos: {corr_rating_revenue:.4f}")
print(f"   Cantidad de votos vs ingresos: {corr_votes_revenue:.4f}")

print(f"\n💡 INTERPRETACIÓN:")
if corr_rating_revenue > 0.3:
    print(f"   SÍ hay correlación moderada entre calificación e ingresos")
    print(f"   Películas mejor calificadas tienden a generar más ingresos")
else:
    print(f"   Correlación débil entre calificación e ingresos")
    print(f"   Calidad no necesariamente se traduce en éxito comercial")

if corr_votes_revenue > 0.5:
    print(f"   FUERTE correlación entre cantidad de votos e ingresos")
    print(f"   Más votos indica mayor audiencia y alcance")
else:
    print(f"   Correlación moderada entre votos e ingresos")

# Categorizar por calificación
df_ratings['rating_category'] = pd.cut(df_ratings['voteAvg'],
                                        bins=[0, 5, 6, 7, 8, 10],
                                        labels=['Muy Mala (<5)', 'Mala (5-6)', 
                                               'Regular (6-7)', 'Buena (7-8)', 
                                               'Excelente (8-10)'])

rating_stats = df_ratings.groupby('rating_category').agg({
    'revenue_millions': ['mean', 'median', 'count']
})

print(f"\n INGRESOS POR CATEGORÍA DE CALIFICACIÓN:\n")
for category in rating_stats.index:
    count = rating_stats.loc[category, ('revenue_millions', 'count')]
    avg = rating_stats.loc[category, ('revenue_millions', 'mean')]
    median = rating_stats.loc[category, ('revenue_millions', 'median')]
    
    print(f"  {category}:")
    print(f"     Ingresos promedio: ${avg:.2f}M")
    print(f"     Ingresos mediana: ${median:.2f}M")
    print(f"     Películas: {int(count):,}\n")

print(" CONCLUSIONES:")
print("   Películas excelentes no siempre son las más taquilleras")
print("   Marketing y timing son tan importantes como la calidad")
print("   Cantidad de votos (engagement) es mejor predictor que calificación")
print("   Balance entre calidad artística y apelación comercial es clave")

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter calificación vs ingresos
axes[0, 0].scatter(df_ratings['voteAvg'], df_ratings['revenue_millions'], 
                   alpha=0.3, s=30, color='steelblue')
axes[0, 0].set_title(f'Calificación vs Ingresos (r={corr_rating_revenue:.3f})', fontweight='bold')
axes[0, 0].set_xlabel('Calificación Promedio', fontweight='bold')
axes[0, 0].set_ylabel('Ingresos (millones USD)')
axes[0, 0].grid(True, alpha=0.3)

# Scatter votos vs ingresos (escala log)
axes[0, 1].scatter(df_ratings['voteCount'], df_ratings['revenue_millions'], 
                   alpha=0.3, s=30, color='coral')
axes[0, 1].set_xscale('log')
axes[0, 1].set_title(f'Cantidad de Votos vs Ingresos (r={corr_votes_revenue:.3f})', fontweight='bold')
axes[0, 1].set_xlabel('Cantidad de Votos (escala log)', fontweight='bold')
axes[0, 1].set_ylabel('Ingresos (millones USD)')
axes[0, 1].grid(True, alpha=0.3)

# Boxplot ingresos por categoría de calificación
df_ratings.boxplot(column='revenue_millions', by='rating_category', ax=axes[1, 0])
axes[1, 0].set_title('Ingresos por Categoría de Calificación', fontweight='bold')
axes[1, 0].set_xlabel('Categoría de Calificación', fontweight='bold')
axes[1, 0].set_ylabel('Ingresos (millones USD)')
axes[1, 0].tick_params(axis='x', rotation=45)
plt.suptitle('')

# Histograma de calificaciones
axes[1, 1].hist(df_ratings['voteAvg'], bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(df_ratings['voteAvg'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {df_ratings["voteAvg"].mean():.2f}')
axes[1, 1].set_title('Distribución de Calificaciones', fontweight='bold')
axes[1, 1].set_xlabel('Calificación', fontweight='bold')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

save_figure('imagenes/parte4_04_calificaciones_vs_exito.png')
plt.close()


# 4.15. ESTRATEGIAS DE MARKETING

print_section("4.15. ESTRATEGIAS DE MARKETING Y RESULTADOS", "·")

df_marketing = df[df['revenue'] > 0].copy()
df_marketing['revenue_millions'] = df_marketing['revenue'] / 1_000_000
df_marketing['has_homepage'] = df_marketing['homePage'].notna() & (df_marketing['homePage'] != "")
df_marketing['has_video'] = df_marketing['video'] == True

# Análisis de homepage
with_hp = df_marketing[df_marketing['has_homepage']]['revenue_millions'].mean()
without_hp = df_marketing[~df_marketing['has_homepage']]['revenue_millions'].mean()
with_hp_median = df_marketing[df_marketing['has_homepage']]['revenue_millions'].median()
without_hp_median = df_marketing[~df_marketing['has_homepage']]['revenue_millions'].median()

# Análisis de video
with_video = df_marketing[df_marketing['has_video']]['revenue_millions'].mean()
without_video = df_marketing[~df_marketing['has_video']]['revenue_millions'].mean()
with_video_median = df_marketing[df_marketing['has_video']]['revenue_millions'].median()
without_video_median = df_marketing[~df_marketing['has_video']]['revenue_millions'].median()

print(f" IMPACTO DE ESTRATEGIAS DE MARKETING:\n")

print(f"📱 PÁGINA OFICIAL (HomePage):")
print(f"  Con homepage:")
print(f"     Ingresos promedio: ${with_hp:.2f}M")
print(f"     Ingresos mediana: ${with_hp_median:.2f}M")
print(f"     Películas: {df_marketing['has_homepage'].sum():,}")
print(f"  Sin homepage:")
print(f"     Ingresos promedio: ${without_hp:.2f}M")
print(f"     Ingresos mediana: ${without_hp_median:.2f}M")
print(f"     Películas: {(~df_marketing['has_homepage']).sum():,}")
diff_hp = ((with_hp - without_hp) / without_hp * 100) if without_hp > 0 else 0
print(f"  📈 Diferencia: {diff_hp:+.1f}% más ingresos con homepage\n")

print(f"🎥 VIDEO PROMOCIONAL:")
print(f"  Con video:")
print(f"     Ingresos promedio: ${with_video:.2f}M")
print(f"     Ingresos mediana: ${with_video_median:.2f}M")
print(f"     Películas: {df_marketing['has_video'].sum():,}")
print(f"  Sin video:")
print(f"     Ingresos promedio: ${without_video:.2f}M")
print(f"     Ingresos mediana: ${without_video_median:.2f}M")
print(f"     Películas: {(~df_marketing['has_video']).sum():,}")
diff_video = ((with_video - without_video) / without_video * 100) if without_video > 0 else 0
print(f"   Diferencia: {diff_video:+.1f}% más ingresos con video\n")

# Combinación de estrategias
df_marketing['marketing_score'] = (df_marketing['has_homepage'].astype(int) + 
                                    df_marketing['has_video'].astype(int))

marketing_impact = df_marketing.groupby('marketing_score')['revenue_millions'].agg(['mean', 'median', 'count'])

print(f" IMPACTO COMBINADO DE ESTRATEGIAS:\n")
marketing_labels = {0: "Sin marketing digital", 1: "Una estrategia", 2: "Ambas estrategias"}
for score, label in marketing_labels.items():
    if score in marketing_impact.index:
        print(f"  {label}:")
        print(f"    • Ingresos promedio: ${marketing_impact.loc[score, 'mean']:.2f}M")
        print(f"    • Ingresos mediana: ${marketing_impact.loc[score, 'median']:.2f}M")
        print(f"    • Películas: {int(marketing_impact.loc[score, 'count']):,}\n")

print(" INTERPRETACIÓN:")
if diff_hp > 20 or diff_video > 20:
    print("  • Marketing digital tiene impacto SIGNIFICATIVO en ingresos")
    print("  • Películas con presencia digital generan notablemente más ingresos")
else:
    print("  • Marketing digital tiene impacto MODERADO en ingresos")
    print("  • Refleja correlación, no necesariamente causalidad")


# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Homepage impacto
axes[0, 0].bar(['Sin HomePage', 'Con HomePage'], [without_hp, with_hp], 
               color=['lightcoral', 'lightgreen'], alpha=0.7)
axes[0, 0].set_title('Impacto de Página Oficial', fontweight='bold')
axes[0, 0].set_ylabel('Ingresos Promedio (millones USD)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Video impacto
axes[0, 1].bar(['Sin Video', 'Con Video'], [without_video, with_video],
               color=['lightcoral', 'lightgreen'], alpha=0.7)
axes[0, 1].set_title('Impacto de Video Promocional', fontweight='bold')
axes[0, 1].set_ylabel('Ingresos Promedio (millones USD)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Marketing combinado
marketing_impact['mean'].plot(kind='bar', ax=axes[1, 0], color='purple', alpha=0.7)
axes[1, 0].set_title('Impacto Combinado de Estrategias', fontweight='bold')
axes[1, 0].set_xlabel('Número de Estrategias')
axes[1, 0].set_ylabel('Ingresos Promedio (millones USD)')
axes[1, 0].set_xticklabels(['Ninguna', 'Una', 'Ambas'], rotation=0)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Distribución
df_marketing['marketing_score'].value_counts().sort_index().plot(kind='bar', 
                                                                  ax=axes[1, 1], 
                                                                  color='teal', alpha=0.7)
axes[1, 1].set_title('Distribución de Estrategias de Marketing', fontweight='bold')
axes[1, 1].set_xlabel('Número de Estrategias')
axes[1, 1].set_ylabel('Cantidad de Películas')
axes[1, 1].set_xticklabels(['Ninguna', 'Una', 'Ambas'], rotation=0)
axes[1, 1].grid(True, alpha=0.3, axis='y')

save_figure('imagenes/parte4_05_estrategias_marketing.png')
plt.close()


# 4.16. POPULARIDAD DEL ELENCO VS ÉXITO DE TAQUILLA

print_section("4.16. POPULARIDAD DEL ELENCO VS ÉXITO DE TAQUILLA", "·")

df_cast_pop = df[(df['revenue'] > 0) & (df['actorsPopularity'].notna())].copy()
df_cast_pop['revenue_millions'] = df_cast_pop['revenue'] / 1_000_000
df_cast_pop['avg_cast_popularity'] = df_cast_pop['actorsPopularity'].apply(parse_popularity)
df_cast_pop = df_cast_pop[df_cast_pop['avg_cast_popularity'].notna()]

# Correlación
correlation_cast = df_cast_pop['avg_cast_popularity'].corr(df_cast_pop['revenue_millions'])

print(f"CORRELACIÓN popularidad elenco vs ingresos: {correlation_cast:.4f}\n")

if correlation_cast > 0.5:
    interpretacion = "FUERTE y POSITIVA"
    conclusion = "SÍ, hay correlación directa considerable"
elif correlation_cast > 0.3:
    interpretacion = "MODERADA y POSITIVA"
    conclusion = "SÍ, existe correlación moderada"
elif correlation_cast > 0.1:
    interpretacion = "DÉBIL pero POSITIVA"
    conclusion = "Correlación débil, no es factor determinante"
else:
    interpretacion = "MUY DÉBIL o NULA"
    conclusion = "NO hay correlación significativa"

print(f"  Interpretación: Correlación {interpretacion}")
print(f"   Conclusión: {conclusion}")

# Categorizar por popularidad
df_cast_pop['pop_category'] = pd.cut(df_cast_pop['avg_cast_popularity'],
                                       bins=[0, 5, 15, 50, 100],
                                       labels=['Baja (0-5)', 'Media (5-15)', 
                                              'Alta (15-50)', 'Muy Alta (50-100)'])

pop_stats = df_cast_pop.groupby('pop_category').agg({
    'revenue_millions': ['mean', 'median', 'count']
})

print(f"\n INGRESOS POR CATEGORÍA DE POPULARIDAD DEL ELENCO:\n")
for category in pop_stats.index:
    count = pop_stats.loc[category, ('revenue_millions', 'count')]
    avg = pop_stats.loc[category, ('revenue_millions', 'mean')]
    median = pop_stats.loc[category, ('revenue_millions', 'median')]
    
    print(f"  {category}:")
    print(f"     Ingresos promedio: ${avg:.2f}M")
    print(f"     Ingresos mediana: ${median:.2f}M")
    print(f"     Películas: {int(count):,}\n")

# Estadísticas generales
print(f"ESTADÍSTICAS GENERALES:")
print(f"   Popularidad promedio del elenco: {df_cast_pop['avg_cast_popularity'].mean():.2f}")
print(f"   Popularidad mediana: {df_cast_pop['avg_cast_popularity'].median():.2f}")
print(f"   Popularidad mínima: {df_cast_pop['avg_cast_popularity'].min():.2f}")
print(f"   Popularidad máxima: {df_cast_pop['avg_cast_popularity'].max():.2f}")

print("\n INTERPRETACIÓN:")
if correlation_cast > 0.3:
    print("   Elencos populares SÍ atraen más audiencia")
    print("   Actores famosos generan expectativa y marketing orgánico")
    print("   Star power es factor comercial importante")
else:
    print("   Popularidad del elenco NO garantiza éxito de taquilla")
    print("   Otros factores son más determinantes:")
    print("     Calidad del guión")
    print("     Dirección")
    print("     Género de la película")
    print("     Marketing y distribución")

print("\n CONCLUSIONES:")
print("   Actores populares facilitan financiamiento y distribución")
print("   No reemplazan necesidad de historia sólida")
print("   Elencos balanceados (estrellas + talento emergente) son efectivos")
print("   Popularidad debe complementar, no definir, decisiones de casting")

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter popularidad vs ingresos
axes[0, 0].scatter(df_cast_pop['avg_cast_popularity'], df_cast_pop['revenue_millions'], 
                   alpha=0.3, s=30, color='steelblue')
axes[0, 0].set_title(f'Popularidad Elenco vs Ingresos (r={correlation_cast:.3f})', 
                     fontweight='bold')
axes[0, 0].set_xlabel('Popularidad Promedio del Elenco', fontweight='bold')
axes[0, 0].set_ylabel('Ingresos (millones USD)')
axes[0, 0].grid(True, alpha=0.3)

# Boxplot ingresos por categoría de popularidad
df_cast_pop.boxplot(column='revenue_millions', by='pop_category', ax=axes[0, 1])
axes[0, 1].set_title('Ingresos por Categoría de Popularidad', fontweight='bold')
axes[0, 1].set_xlabel('Categoría de Popularidad', fontweight='bold')
axes[0, 1].set_ylabel('Ingresos (millones USD)')
axes[0, 1].tick_params(axis='x', rotation=45)
plt.suptitle('')

# Histograma de popularidad
axes[1, 0].hist(df_cast_pop['avg_cast_popularity'], bins=30, color='coral', 
                alpha=0.7, edgecolor='black')
axes[1, 0].axvline(df_cast_pop['avg_cast_popularity'].mean(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Media: {df_cast_pop["avg_cast_popularity"].mean():.1f}')
axes[1, 0].set_title('Distribución de Popularidad del Elenco', fontweight='bold')
axes[1, 0].set_xlabel('Popularidad Promedio', fontweight='bold')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Barras promedio por categoría
pop_stats[('revenue_millions', 'mean')].plot(kind='bar', ax=axes[1, 1], color='green', alpha=0.7)
axes[1, 1].set_title('Ingresos Promedio por Categoría', fontweight='bold')
axes[1, 1].set_xlabel('Categoría de Popularidad', fontweight='bold')
axes[1, 1].set_ylabel('Ingresos Promedio (millones USD)')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

save_figure('imagenes/parte4_06_popularidad_elenco_vs_taquilla.png')
plt.close()





import pandas as pd
import numpy as np

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


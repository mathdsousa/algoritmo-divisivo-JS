import pandas as pd
import numpy as np
from KMeans import KMeans

# Criando um DataFrame com 10 objetos e 4 atributos intervalares
np.random.seed(42)  # Garantir reprodutibilidade
data = {
    "Objeto": [f"Objeto_{i+1}" for i in range(10)],
    "Atributo_1": np.random.uniform(10, 20, 10),
    "Atributo_2": np.random.uniform(30, 50, 10),
    "Atributo_3": np.random.uniform(5, 15, 10),
    "Atributo_4": np.random.uniform(100, 200, 10),
}

df = pd.DataFrame(data)
print(df)

# Teste do algoritmo MST_DivisiveClustering
KMeans = KMeans(num_clusters=3, init='random')
df['Cluster'] = KMeans.fit_predict(df.drop(columns='Objeto'))
print(df)

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from MST_DivisiveClustering import MST_DivisiveClustering

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
MST_DivisiveClustering = MST_DivisiveClustering(n_clusters=3, metric='euclidean')
MST_DivisiveClustering.fit(df.drop(columns='Objeto'))

# Teste do algoritmo AgglomerativeClustering (para comparação)
AgglomerativeClustering = AgglomerativeClustering(n_clusters=3, linkage='single')
AgglomerativeClustering.fit(df.drop(columns='Objeto'))

print("Labels do MST_DivisiveClustering:")
print(MST_DivisiveClustering.labels_)
print("Labels do AgglomerativeClustering:")
print(AgglomerativeClustering.labels_)



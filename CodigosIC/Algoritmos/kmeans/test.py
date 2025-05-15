from cProfile import label
import pandas as pd
import numpy as np
from kMeans import KMeans as kMedias
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

print("Iniciando o teste do algoritmo kMeans")    

# Criando um DataFrame com 10 objetos e 4 atributos intervalares
data = {
    "Objeto": [f"Objeto_{i+1}" for i in range(10)],
    "Atributo_1": np.random.uniform(10, 20, 10),
    "Atributo_2": np.random.uniform(30, 50, 10),
    "Atributo_3": np.random.uniform(5, 15, 10),
    "Atributo_4": np.random.uniform(100, 200, 10),
}

df = pd.DataFrame(data)

# Teste do algoritmo kMeans
kMeans = kMedias(num_clusters=3, init='random', max_iter=200)
kMeans.fit(df.drop(columns='Objeto'))

# Usando a scilit-learn para comparação - pura curiosidade
kMeans_sklearn = KMeans(n_clusters=3, n_init=1, max_iter=200)
kMeans_sklearn.fit(df.drop(columns='Objeto'))

print("Labels do kMeans (implementação própria):")
print(kMeans.labels_)   
print("Centros do kMeans (implementação própria):")
print(kMeans.cluster_centers_)
print("Labels do kMeans (scikit-learn):")   
print(kMeans_sklearn.labels_)
print("Centros do kMeans (scikit-learn):")
print(kMeans_sklearn.cluster_centers_)

import pandas as pd

data_v4 = {
    "Rand index": [0.510262, 0.643175, 0.643175, 0.547213],
    "Adjusted Rand index": [-0.015430, -0.004157, -0.004157, 0.054028],
    "Mutual info score": [0.005668, 0.000771, 0.000771, 0.022123],
    "Adjusted mutual info score": [0.007629, -0.002987, -0.002987, 0.027204],
    "Fowlkes Mallows index": [0.592545, 0.800673, 0.800673, 0.629800],
    "Homogeneity score": [0.010547, 0.001435, 0.001435, 0.041164],
    "Completeness score": [0.012879, 0.038108, 0.038108, 0.030347],
    "V measure": [0.010338, 0.002765, 0.002765, 0.034937],
    "Silhouette coefficient": [0.181733, 0.643707, 0.643707, 0.086263],
    "Calinski Harabasz score": [56.625034, 13.810964, 13.810964, 13.223436],
    "Davies Bouldin score": [2.140624, 0.255589, 0.255589, 2.491329]
}

# Nomes dos algoritmos
algorithms_v4 = [
    "KMeans",
    "MST - Euclidean",
    "MST - Jensen-Shannon",
    "HDBSCAN"
]

# Criar DataFrame
df_v4 = pd.DataFrame(data_v4, index=algorithms_v4)

# Salvar em arquivo Excel
file_path_v4 = "./indices_agrupamento.xlsx"
df_v4.to_excel(file_path_v4)

file_path_v4
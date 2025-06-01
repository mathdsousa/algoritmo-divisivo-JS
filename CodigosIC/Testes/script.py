import pandas as pd

data = {
    "Rand index": [0.878684, 0.103996, 0.103965, 0.626892],
    "Adjusted Rand index": [0.369060, 0.000004, -0.000031, 0.007895],
    "Mutual info score": [1.198695, 0.005931, 0.005853, 0.897422],
    "Adjusted mutual info score": [0.527740, 0.000008, -0.000060, 0.175272],
    "Fowlkes Mallows index": [0.437835, 0.315234, 0.315184, 0.193666],
    "Homogeneity score": [0.520742, 0.002577, 0.002543, 0.389862],
    "Completeness score": [0.540043, 0.251831, 0.248517, 0.285465],
    "V measure": [0.530167, 0.005101, 0.005034, 0.329595],
    "Silhouette coefficient": [0.142664, 0.030805, -0.054370, -0.248589],
    "Calinski Harabasz score": [434.335461, 1.803401, 1.255965, 3.923444],
    "Davies Bouldin score": [1.962363, 0.748400, 0.890673, 1.420626],
}

# Nomes dos algoritmos
algorithms_v4 = [
    "KMeans",
    "MST - Euclidean",
    "MST - Jensen-Shannon",
    "HDBSCAN"
]

# Criar DataFrame
df_v4 = pd.DataFrame(data, index=algorithms_v4)

# Salvar em arquivo Excel
file_path_v4 = "./indices_agrupamento.xlsx"
df_v4.to_excel(file_path_v4)

file_path_v4
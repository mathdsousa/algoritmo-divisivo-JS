import pandas
import scikit_posthocs as sp
from scipy import stats

############################# Teste de Hipótese #############################
df = pandas.read_csv('./Metricas/silhouette.csv')
# Converte para matriz (numpy)
dados = df.to_numpy()

print(dados)
print()

# Separa os grupos
kmeans = dados[:, 0]
mst_euclidean = dados[:, 1]
mst_js = dados[:, 2]
hdbscan = dados[:, 3]

# Realiza o teste de Friedman
print(stats.friedmanchisquare(kmeans, mst_euclidean, mst_js, hdbscan))
print()

# Realiza o teste posthoc de Nemenyi
print(sp.posthoc_nemenyi_friedman(dados.astype(float)))
print()

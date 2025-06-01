# Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import warnings
import networkx as nx
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
import scipy.sparse._csr
import scipy
import scipy.spatial.distance as SSD
from Algoritmos.mst_DivisiveClustering.MST_DivisiveClustering import MST_DivisiveClustering
from numpy import dot
from numpy import trace
from numpy.linalg import inv
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Implementação própria do algoritmo K-means
def Kmeans(dados, target):
    print('Execução do K-means...')
    iteracao = 0
    # Número de amostras, atributos e clusters
    n = dados.shape[0]
    m = dados.shape[1]
    k = len(np.unique(target))
    # Escolhe k centros aleatórios
    centros = dados[np.random.choice(n, size=k, replace=False)]
    # Cria vetor de rótulos
    rotulos = np.zeros(n)
    # Loop principal
    while True:
        iteracao += 1
        # Calcula as distâncias de cada amostra para os k centros
        distancias = np.zeros(k)
        for i in range(n):
            for j in range(centros.shape[0]):
                distancias[j] = np.linalg.norm(dados[i, :] - centros[j, :])
            rotulos[i] = distancias.argmin()
        # Atualiza os centros
        novos_centros = np.zeros((k, m))
        for r in range(k):
            indices = np.where(rotulos==r)[0]
            if len(indices) > 0:
                amostras = dados[indices]
                novos_centros[r, :] = amostras.mean(axis=0)
            else:
                novos_centros[r, :] = centros[r, :]
        # Verifica condição de parada
        if (np.linalg.norm(centros - novos_centros) < 1) or iteracao > 10:
            break
        # Atualiza centros
        centros = novos_centros.copy()

    return rotulos


# Extrai a MST do grafo completo gerado a partir das amostras (distância Euclidiana e divergência de Jensen-Shannon)
def extrai_MST(dados):
    n = dados.shape[0]
    m = dados.shape[1]
    k = round(np.sqrt(n))
    # Gera o grafo completo
    Kn = sknn.kneighbors_graph(dados, n_neighbors=n-1, mode='distance', include_self=False)
    W = Kn.toarray()
    # Extrai MST baseada na distância Euclidiana
    G = nx.from_numpy_array(W)
    # Euclidean distance based MST
    T_euc = nx.minimum_spanning_tree(G)
    # Divergência de Jensen-Shannon
    W = SSD.cdist(dados, dados, metric='jensenshannon')
    # Cria o grafo completo com divergência de JS
    H = nx.from_numpy_array(W)
    # JS divergence based MST
    T_js = nx.minimum_spanning_tree(H)
    
    return T_euc, T_js


# MST based clustering: divisive approach (Calinski-Harabasz)
def mst_clustering_divisive_CH(T, dados, rotulos):
    # Número de amostras
    n = len(rotulos)
    # Número de classes
    c = len(np.unique(rotulos))
    # Número de arestas a ser removidas
    K = c - 1
    # Número de arestas na MST
    n_edges = len(T.edges())
    # Loop principal
    T_ = T.copy()
    for i in range(K):
        # Vetor de medidas (sc, ch, db, etc)
        arestas = []
        medidas = []
        # Encontrar aresta cuja remoção maximize o índice CH
        for (u, v) in T_.edges():
            arestas.append((u, v))
            labels = np.zeros(n)
            T_.remove_edge(u, v)
            clusters = nx.connected_components(T_)
            code = 0
            arvores = []
            for c in clusters:  # c is a set
                indices = np.array(list(c))                
                labels[indices] = code
                code += 1
            ch = calinski_harabasz_score(dados, labels)
            medidas.append(ch)             
            T_.add_edge(u, v)
        best = np.array(medidas).argmax()
        edge_star = arestas[best]
        T_.remove_edge(*edge_star)
    return T_

# Cria rótulos
def rotula_amostras(T, rotulos):
    # Obtém os componentes conexos 
    n = len(rotulos)
    clusters = nx.connected_components(T)
    labels = np.zeros(n)
    code = 0
    for c in clusters:  # c is a set
        indices = np.array(list(c))
        labels[indices] = code
        code += 1
    return labels

########################################################################
# INÍCIO DO SCRIPT
########################################################################

#%%%%%%%%%%%%%%%%%%%%  Data loading
# Descomentar apenas uma linha por vez!
#X = skdata.load_iris()
#X = skdata.load_wine()
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='AP_Lung_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Kidney', version=1)       
#X = skdata.fetch_openml(name='AP_Colon_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Colon', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Omentum', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Omentum', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Prostate', version=1)
#X = skdata.fetch_openml(name='AP_Lung_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Prostate', version=1)
#X = skdata.fetch_openml(name='parkinson-speech-uci', version=1)
X = skdata.fetch_openml(name='semeion', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='Olivetti_Faces', version=1)
#X = skdata.fetch_openml(name='oh5.wc', version=1)       
#X = skdata.fetch_openml(name='dbworld-bodies', version=1)     
#X = skdata.fetch_openml(name='oh15.wc', version=1)     
#X = skdata.fetch_openml(name='tr31.wc', version=1)     
#X = skdata.fetch_openml(name='BurkittLymphoma', version=1)     
#X = skdata.fetch_openml(name='ovarianTumour', version=1)
#X = skdata.fetch_openml(name='hepatitisC', version=1)
#X = skdata.fetch_openml(name='micro-mass', version=2)
#X = skdata.fetch_openml(name='scene', version=1)
#X = skdata.fetch_openml(name='musk', version=1)
#X = skdata.fetch_openml(name='Speech', version=1)
#X = skdata.fetch_openml(name='MNIST_784', version=1)    # 5% das amostras
#X = skdata.fetch_openml(name='Fashion-MNIST', version=1)    # 5% das amostras

dados = X['data']
target = X['target']  

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

# Matriz esparsa (em alguns datasets com dimensionalidade muito alta)
if type(dados) == scipy.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
        target = target.to_numpy()

dados = np.nan_to_num(dados)

# Transforma rótulos para inteiros
rotulos = list(np.unique(target))
numbers = np.zeros(n)
for i in range(n):
    numbers[i] = rotulos.index(target[i])
target = numbers.copy()

# MNIST e Fashion-MNIST: reduz número de amostras
if 'details' in X.keys():
    if X['details']['name'] == 'mnist_784' or X['details']['name'] == 'Fashion-MNIST':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.05, random_state=42)
    
n = dados.shape[0]

print('N = ', n)
print('M = ', m)
print('C = %d' %c)


# print('MST')
# # Extrai MST's (Euclidiana e Jensen-Shannon)
# MST_euc, MST_js = extrai_MST(dados)

# print('Euclidean e JS distance based MST')
# # MST based clustering with CH index
# t = mst_clustering_divisive_CH(MST_euc, dados, target)
# tree = mst_clustering_divisive_CH(MST_js, dados, target)

# # Rotula as amostras
# labels = rotula_amostras(t, target)        # CH (Euclidean)
# labs = rotula_amostras(tree, target)       # CH (JS divergence)

# # Minha implementação
# print('Iniciando o teste do algoritmo MST_DivisiveClustering_euclidean')
# MST_Div_Euclidean = MST_DivisiveClustering(n_clusters=c, metric='euclidean')
# MST_Div_Euclidean.fit(dados)

# print(f'São iguais? {np.array_equal(labels, MST_Div_Euclidean.labels_)}')

print('KMeans')
# K-médias (como depende da inicialização, executamos várias vezes)
MAX = 20
L_rand = []
L_ari = []
L_mi = []
L_ami = []
L_fm = []
L_hom = []
L_comp = []
L_vm = []
L_sc = []
L_ch = []
L_db = []
for i in range(MAX):
    kmeanslabels = Kmeans(dados, target)
    rand = rand_score(target, kmeanslabels)
    L_rand.append(rand)
    ari = adjusted_rand_score(target, kmeanslabels)
    L_ari.append(ari)
    mi = mutual_info_score(target, kmeanslabels)
    L_mi.append(mi)
    ami = adjusted_mutual_info_score(target, kmeanslabels)
    L_ami.append(ami)
    fm = fowlkes_mallows_score(target, kmeanslabels)
    L_fm.append(fm)
    hom = homogeneity_score(target, kmeanslabels)
    L_hom.append(hom)
    comp = completeness_score(target, kmeanslabels)
    L_comp.append(comp)
    vm = v_measure_score(target, kmeanslabels)
    L_vm.append(vm)
    sc = silhouette_score(dados, kmeanslabels)
    L_sc.append(sc)
    ch = calinski_harabasz_score(dados, kmeanslabels)
    L_ch.append(ch)
    db = davies_bouldin_score(dados, kmeanslabels)
    L_db.append(db)

# Calcula as médias
M_rand = np.array(L_rand).mean()
M_ari = np.array(L_ari).mean()
M_mi = np.array(L_mi).mean()
M_ami = np.array(L_ami).mean()
M_fm = np.array(L_fm).mean()
M_hom = np.array(L_hom).mean()
M_comp = np.array(L_comp).mean()
M_vm = np.array(L_vm).mean()
M_sc = np.array(L_sc).mean()
M_ch = np.array(L_ch).mean()
M_db = np.array(L_db).mean()

print()
print('True labels:', target)
print()
print('Kmeans labels: ', kmeanslabels)
print()
print('Euclidean distance based MST labels: ', labels)
print()
print('Jensen-Shannon divergence based MST labels: ', labs)
print()


####### External indices
print('Kmeans indices')
print('------------------------')
print('Rand index: %f' %M_rand)
print('Adjusted Rand index: %f' %M_ari)
print('Mutual info score: %f' %M_mi)
print('Adjusted mutual info score: %f' %M_ami)
print('Fowlkes Mallows index: %f' %M_fm)
print('Homogeneity score: %f' %M_hom)
print('Completeness score: %f' %M_comp)
print('V measure: %f' %M_vm)
print('Silhouette coefficient: %f' %M_sc)
print('Calinski Harabasz score: %f' %M_ch)
print('Davies Bouldin score: %f' %M_db)
print()

print('Euclidean distance MST-based indices')
print('---------------------------------------')
print('Rand index: %f' %rand_score(target, labels))
print('Adjusted Rand index: %f' %adjusted_rand_score(target, labels))
print('Mutual info score: %f' %mutual_info_score(target, labels))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, labels))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, labels))
print('Homogeneity score: %f' %homogeneity_score(target, labels))
print('Completeness score: %f' %completeness_score(target, labels))
print('V measure: %f' %v_measure_score(target, labels))
print('Silhouette coefficient: %f' %silhouette_score(dados, labels))
print('Calinski Harabasz score: %f' %calinski_harabasz_score(dados, labels))
print('Davies Bouldin score: %f' %davies_bouldin_score(dados, labels))
print()

print('Jensen-Shannon divergence MST-based indices')
print('---------------------------------------------')
print('Rand index: %f' %rand_score(target, labs))
print('Adjusted Rand index: %f' %adjusted_rand_score(target, labs))
print('Mutual info score: %f' %mutual_info_score(target, labs))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, labs))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, labs))
print('Homogeneity score: %f' %homogeneity_score(target, labs))
print('Completeness score: %f' %completeness_score(target, labs))
print('V measure: %f' %v_measure_score(target, labs))
print('Silhouette coefficient: %f' %silhouette_score(dados, labs))
print('Calinski Harabasz score: %f' %calinski_harabasz_score(dados, labels))
print('Davies Bouldin score: %f' %davies_bouldin_score(dados, labs))
print()
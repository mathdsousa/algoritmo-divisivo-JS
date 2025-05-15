# Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sklearn.datasets as skdata
import numpy as np
import scipy.sparse._csr
import scipy
from Algoritmos.kmeans.kMeans import KMeans 
from Algoritmos.mst_DivisiveClustering.MST_DivisiveClustering import MST_DivisiveClustering
from sklearn.cluster import HDBSCAN
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################
# INÍCIO DO SCRIPT
########################################################################

print("Iniciando os testes")

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
X = skdata.fetch_openml(name='AP_Omentum_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Prostate', version=1)
#X = skdata.fetch_openml(name='AP_Lung_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Prostate', version=1)
#X = skdata.fetch_openml(name='parkinson-speech-uci', version=1)
#X = skdata.fetch_openml(name='semeion', version=1)
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
print()

print('Iniciando o teste do algoritmo MST_DivisiveClustering_euclidean')
MST_Div_Euclidean = MST_DivisiveClustering(n_clusters=c, metric='euclidean')
MST_Div_Euclidean.fit(dados)

print('Iniciando o teste do algoritmo MST_DivisiveClustering_jensenshannon')
MST_Div_JS = MST_DivisiveClustering(n_clusters=c, metric='jensenshannon')
MST_Div_JS.fit(dados)

print('Iniciando o teste do algoritmo HDBSCAN')
hdb = HDBSCAN(min_samples=3)
hdb.fit(dados)

print('Iniciando o teste do algoritmo KMeans')
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
    kmedias = KMeans(num_clusters=c, init='random', max_iter=200)
    kmedias.fit(dados)
    rand = rand_score(target, kmedias.labels_)
    L_rand.append(rand)
    ari = adjusted_rand_score(target, kmedias.labels_)
    L_ari.append(ari)
    mi = mutual_info_score(target, kmedias.labels_)
    L_mi.append(mi)
    ami = adjusted_mutual_info_score(target, kmedias.labels_)
    L_ami.append(ami)
    fm = fowlkes_mallows_score(target, kmedias.labels_)
    L_fm.append(fm)
    hom = homogeneity_score(target, kmedias.labels_)
    L_hom.append(hom)
    comp = completeness_score(target, kmedias.labels_)
    L_comp.append(comp)
    vm = v_measure_score(target, kmedias.labels_)
    L_vm.append(vm)
    sc = silhouette_score(dados, kmedias.labels_)
    L_sc.append(sc)
    ch = calinski_harabasz_score(dados, kmedias.labels_)
    L_ch.append(ch)
    db = davies_bouldin_score(dados, kmedias.labels_)
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
print('Labels:', target)
print()
print('Kmeans labels: ', kmedias.labels_)
print()
print('Euclidean distance based MST labels: ', MST_Div_Euclidean.labels_)
print()
print('Jensen-Shannon divergence based MST labels: ', MST_Div_JS.labels_)
print()
print('HDBSCAN labels: ', hdb.labels_)
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
print('Rand index: %f' %rand_score(target, MST_Div_Euclidean.labels_))
print('Adjusted Rand index: %f' %adjusted_rand_score(target, MST_Div_Euclidean.labels_))
print('Mutual info score: %f' %mutual_info_score(target, MST_Div_Euclidean.labels_))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, MST_Div_Euclidean.labels_))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, MST_Div_Euclidean.labels_))
print('Homogeneity score: %f' %homogeneity_score(target, MST_Div_Euclidean.labels_))
print('Completeness score: %f' %completeness_score(target, MST_Div_Euclidean.labels_))
print('V measure: %f' %v_measure_score(target, MST_Div_Euclidean.labels_))
print('Silhouette coefficient: %f' %silhouette_score(dados, MST_Div_Euclidean.labels_))
print('Calinski Harabasz score: %f' %calinski_harabasz_score(dados, MST_Div_Euclidean.labels_))
print('Davies Bouldin score: %f' %davies_bouldin_score(dados, MST_Div_Euclidean.labels_))
print()

print('Jensen-Shannon divergence MST-based indices')
print('---------------------------------------------')
print('Rand index: %f' %rand_score(target, MST_Div_JS.labels_))
print('Adjusted Rand index: %f' %adjusted_rand_score(target, MST_Div_JS.labels_))
print('Mutual info score: %f' %mutual_info_score(target, MST_Div_JS.labels_))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, MST_Div_JS.labels_))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, MST_Div_JS.labels_))
print('Homogeneity score: %f' %homogeneity_score(target, MST_Div_JS.labels_))
print('Completeness score: %f' %completeness_score(target, MST_Div_JS.labels_))
print('V measure: %f' %v_measure_score(target, MST_Div_JS.labels_))
print('Silhouette coefficient: %f' %silhouette_score(dados, MST_Div_JS.labels_))
print('Calinski Harabasz score: %f' %calinski_harabasz_score(dados, MST_Div_JS.labels_))
print('Davies Bouldin score: %f' %davies_bouldin_score(dados, MST_Div_JS.labels_))
print()

print('HDBSCAN indices')
print('---------------------------------------------')
print('Rand index: %f' %rand_score(target, hdb.labels_))
print('Adjusted Rand index: %f' %adjusted_rand_score(target, hdb.labels_))
print('Mutual info score: %f' %mutual_info_score(target, hdb.labels_))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, hdb.labels_))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, hdb.labels_))
print('Homogeneity score: %f' %homogeneity_score(target, hdb.labels_))
print('Completeness score: %f' %completeness_score(target, hdb.labels_))
print('V measure: %f' %v_measure_score(target, hdb.labels_))
print('Silhouette coefficient: %f' %silhouette_score(dados, hdb.labels_))
print('Calinski Harabasz score: %f' %calinski_harabasz_score(dados, hdb.labels_))
print('Davies Bouldin score: %f' %davies_bouldin_score(dados, hdb.labels_))
print()
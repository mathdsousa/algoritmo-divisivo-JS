import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jensenshannon
from mstkruskal import MST_Kruskal

def plot_graph(graph, weights=False):
    if weights:
        pos = nx.spring_layout(graph)  

        plt.figure(figsize=(8, 6))
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=12, edge_color='gray')
        edge_labels = nx.get_edge_attributes(graph, 'weight')  # Obtendo os pesos das arestas
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

        plt.title("Grafo com Pesos nas Arestas")
        plt.show()

    plt.figure(figsize=(8, 6))  # Tamanho da figura
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=1500, font_size=12, edge_color='gray')
    plt.title("Exemplo de Grafo com NetworkX")
    plt.show()

def _complete_graph(X, metric):
    n_objects = X.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_objects))

    if (metric == 'euclidean'):
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                graph.add_edge(i, j, weight=euclidean(X[i], X[j]))
                #print(euclidean(X[i], X[j]))

    elif(metric == 'jensenshannon'):
        for i in range(n_objects):
            for j in range(i+1, n_objects):
                graph.add_edge(i, j, weight=jensenshannon(X[i], X[j]))
                #print(jensenshannon(X[i], X[j]))

    return graph

class MST_DivisiveClustering:

    def __init__(self, n_clusters=2, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric

    def fit_predict(self, X):
        # Verifying whether X is an np.ndarray
        if not isinstance(X, np.ndarray):
            X = X.values # Convert the DataFrame to NumPy

        # Obtain the Complete Graph of the data
        graph = _complete_graph(X, self.metric)

        # Obtain the MST    
        mst = MST_Kruskal(graph)
        graph_mst = mst.getMST()

        # Realize k divisions of the edges
        edgesList = graph_mst.edges.data('weight')

        # Reverse sort in the edges 
        sortedEdges = sorted(edgesList, key=lambda x: x[2], reverse=True)

        removedEdges = sortedEdges[:self.n_clusters - 1]

        # Remove k-1 edges (Obtain k clusters)
        graph_mst.remove_edges_from(removedEdges)

        # Obtain Components (Clusters)
        components = nx.connected_components(graph_mst)

        clusters = np.zeros(X.shape[0]) 
        cluster_count = 0
        for component in components:
            for i in component:
                clusters[i] = cluster_count
            cluster_count +=1

        return clusters

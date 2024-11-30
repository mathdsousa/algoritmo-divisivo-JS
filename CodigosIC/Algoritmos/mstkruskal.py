import networkx as nx
from unionfind import UnionFind

class MST_Kruskal:
    def __init__(self, graph) -> None:
        self.graph = graph
    
    def getMST(self) -> nx.Graph:
        R = nx.Graph()
        edgeList = self.graph.edges.data('weight')
        sets = UnionFind()
        sets.MakeUnionFind(self.graph.number_of_nodes())
        edgedSorted = sorted(edgeList, key=lambda x: x[2])

        for vertex1, vertex2, weight_edge in edgedSorted:
            if sets.Find(vertex1) != sets.Find(vertex2):
                sets.Union(vertex1,vertex2)
                R.add_edge(vertex1,  vertex2, weight = weight_edge)
        
        return R
        
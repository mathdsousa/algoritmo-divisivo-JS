from set import Set
from unionfind import UnionFind

def main():
    componentes = UnionFind()

    componentes.MakeUnionFind(10)

    print(componentes.Find(9))
    print(componentes.Find(0))

    componentes.Union(9, 0)

    print(componentes.Find(9))
    print(componentes.Find(0))

    componentes.Union(9, 1)

    print(componentes.Find(9))
    print(componentes.Find(1))

    print(componentes.sets[0].height)


main()
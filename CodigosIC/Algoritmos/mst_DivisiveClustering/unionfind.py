from set import Set

class UnionFind:
    def __init__(self) -> None:
        self.sets = []

    def MakeUnionFind(self, n_vertices) -> None:
        self.sets = [Set(i) for i in range(n_vertices)]

    def FindR(self, obj) -> Set:
        if obj.father is None:
            return obj
        obj.father = self.FindR(obj.father) 
        return obj.father
    
    def Find(self, value) -> int:
        return self.FindR(self.sets[value]).value

    def Union(self, point1, point2) -> None:
        x = self.Find(point1)
        y = self.Find(point2)

        if x != y:
            if self.sets[x].height > self.sets[y].height:
                self.sets[y].father = self.sets[x]
            elif self.sets[x].height < self.sets[y].height:
                self.sets[x].father = self.sets[y]
            else:
                self.sets[y].father = self.sets[x]
                self.sets[x].height += 1 
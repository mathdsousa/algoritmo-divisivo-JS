from set import Set

class UnionFind:
    def __init__(self) -> None:
        self.sets = []

    def MakeUnionFind(self, n_vertices) -> None:
        self.sets = [Set(i) for i in range(n_vertices)]

    def FindR(self, obj) -> Set:
        if(obj.father == None):
            return obj.value
        obj = obj.father
        return self.FindR(obj)
    
    def Find(self, value) -> Set:
        return self.FindR(self.sets[value])
    # def Height(self, point) -> int: 
    #     counter = 0
    #     father = self.sets[point].father
    #     while(father != None):
    #         counter += 1
    #         father = self.sets[point].father
    #     return counter

    def Union(self, point1, point2) -> None:
        x = self.Find(point1)
        y = self.Find(point2)

        if x != y:
            if self.sets[x].height > self.sets[y].height:
                self.sets[y].father = self.sets[x]
            elif self.sets[x].height < self.sets[y].height:
                self.sets[x].father = self.sets[y]
            else:
                self.sets[x].father = self.sets[y]
                self.sets[y].height += 1 
                
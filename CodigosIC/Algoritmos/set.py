class Set:
    def __init__(self, value) -> None:
        self.value = value
        self.father = None
        self.height = 0
        
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, new_father):
        self._father = new_father

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, new_height):
        self._height = new_height
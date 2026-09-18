"""Port of src/GValueStorage.java -- stores values of g(v), a function that
returns geodesic distance from base point b to all points v in the mesh."""


class GValueStorage:
    __slots__ = ("index", "value")

    def __init__(self, index=0, value=0.0):
        self.index = index
        self.value = value

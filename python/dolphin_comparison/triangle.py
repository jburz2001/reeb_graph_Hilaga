"""Port of src/Triangle.java -- properties of a triangle in 3D space."""


class Triangle:
    __slots__ = ("a", "b", "c")

    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c

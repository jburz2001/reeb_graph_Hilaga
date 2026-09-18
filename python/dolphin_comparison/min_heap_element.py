"""Port of src/MinHeapElement.java."""


class MinHeapElement:
    __slots__ = ("index", "key")

    def __init__(self, index=0, key=0.0):
        self.index = index
        self.key = key

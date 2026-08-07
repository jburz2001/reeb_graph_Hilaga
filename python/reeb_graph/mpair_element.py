"""Port of src/MPairElement.java."""


class MPairElement:
    __slots__ = ("node1", "node2", "vector_nodes1", "vector_nodes2")

    def __init__(self):
        self.node1 = None  # RNode from MRG1
        self.node2 = None  # RNode from MRG2
        self.vector_nodes1 = None
        self.vector_nodes2 = None

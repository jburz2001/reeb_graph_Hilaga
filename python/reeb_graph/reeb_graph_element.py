"""Port of src/ReebGraphElement.java -- element of a Reeb graph, used by
MRGConstrLight / ExtractReebGraph while the MRG is being built."""


class ReebGraphElement:
    __slots__ = ("index", "left_bound", "right_bound", "parents", "Tsets")

    def __init__(self):
        self.index = 0
        self.left_bound = 0.0
        self.right_bound = 0.0
        self.parents = None  # list[int] or None
        self.Tsets = None  # list[int] -- indices into all_Tsets

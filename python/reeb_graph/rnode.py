"""Port of src/RNode.java -- R-node used by CompareReebGraph while matching
a pair of MRGs.

NOTE: like the Java class (which never overrides Object.equals /
hashCode), RNode intentionally uses Python's default identity-based
equality -- several parts of the matching algorithm rely on
"the same object" semantics (Vector.contains(...) / Object.equals in
Java == "is"/"in" by identity in Python for a class with no __eq__).
"""


class RNode:
    __slots__ = ("index", "left_bound", "right_bound", "children", "parent", "Tsets", "attribute", "MLIST")

    def __init__(self):
        self.index = 0
        self.left_bound = 0.0
        self.right_bound = 0.0
        self.children = None  # list[int] or None
        self.parent = None  # RNode or None
        self.Tsets = None  # list[int]
        self.attribute = None  # AttributeElement
        self.MLIST = None  # list[int]

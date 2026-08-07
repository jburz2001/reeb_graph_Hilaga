"""Port of src/AttributeElement.java -- attributes held for each MRG node."""


class AttributeElement:
    __slots__ = ("a", "l")

    def __init__(self, a=0.0, l=0.0):
        self.a = a
        self.l = l

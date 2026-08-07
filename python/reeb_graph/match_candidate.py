"""Port of src/MatchCandidate.java.

MatchCandidate maintains a list of R-Node (indices) from one MRG that are
selected as possible candidates for matching with R-Nodes from the other
MRG.
"""


class MatchCandidate:
    __slots__ = ("vector",)

    def __init__(self):
        self.vector = None  # list[int] or None

    def clone_element(self, element):
        self.vector = list(element.vector)

    def add_to_element(self, element):
        self.vector = add_integer_vector(self.vector, element.vector)


def add_integer_vector(target, source):
    """Returns the union of ``target`` and ``source`` (order preserving,
    duplicates skipped), matching MatchCandidate.addIntegerVector."""
    result = None

    if target is None and source is not None:
        result = []
    elif target is not None:
        result = list(target)

    if source is not None:
        for temp in source:
            if temp not in result:
                result.append(temp)

    return result

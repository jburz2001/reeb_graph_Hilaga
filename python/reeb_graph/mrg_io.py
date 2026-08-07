"""Port of src/SaveInText.java (``save_mrg``) and the ``readOneFile`` method
of src/CompareReebGraph.java (``load_mrg``).

Together these two functions define the ``.mrg`` text file format used to
hand off a constructed MRG from ExtractReebGraph to CompareReebGraph. The
schema (block markers, line order) is unchanged from the original Java
version; only the textual representation of floating point numbers
differs (Python's round-trip ``repr()`` vs. Java's ``Double.toString()``)
-- both are shortest round-trip decimal representations that parse back to
the exact same IEEE-754 double, so this has no effect on computed results.
"""

from .attribute_element import AttributeElement
from .java_fmt import java_double_str
from .rnode import RNode


def _fmt_double(x):
    return java_double_str(x)


def save_mrg(filename, MRG, reebs, attributes, res_num):
    """Port of SaveInText.saveIt. ``res_num`` is accepted for interface
    parity with the Java signature but -- exactly as in the original -- is
    never actually used."""
    if ".wrl" in filename:
        graph_filename = filename[: filename.index(".wrl")] + ".mrg"
    else:
        graph_filename = filename + ".mrg"

    with open(graph_filename, "w") as out:
        out.write("attributes{\n")
        out.write(str(len(attributes)) + "\n")

        for element in attributes:
            out.write(_fmt_double(element.a) + " " + _fmt_double(element.l) + "\n")

        out.write("}\n")

        # saving size of MRG
        out.write(str(len(MRG)) + "\n")

        for i in range(len(MRG)):
            out.write("elements{\n")
            out.write(str(len(reebs[i])) + "\n")

            for j in range(len(reebs[i])):
                el = reebs[i][j]

                out.write(str(el.index) + "\n")
                out.write(_fmt_double(el.left_bound) + " " + _fmt_double(el.right_bound) + "\n")

                out.write("".join(str(t) + " " for t in el.Tsets) + "\n")

                if el.parents is not None:
                    out.write("".join(str(p) + " " for p in el.parents) + "\n")
                else:
                    out.write("NULL\n")

            out.write("}\n")

            out.write("connectivity{\n")
            out.write(str(len(MRG[i])) + "\n")

            for j in range(len(MRG[i])):
                row = MRG[i][j]
                line = str(j) + " "
                for k in range(1, row[0]):
                    line += str(row[k]) + " "
                out.write(line + "\n")

            out.write("}\n")


class MrgFormatError(IOError):
    pass


def load_mrg(filename):
    """Port of CompareReebGraph.readOneFile. Returns ``(attributes, mrg)``
    where ``attributes`` is a list[AttributeElement] for the finest
    resolution, and ``mrg`` is ``mrg[res][node_index] -> [self, *neighbors]``
    (a list of RNode, mirroring the nested-Vector structure in the Java
    version so the rest of the matching algorithm can be a direct port)."""
    if ".wrl" in filename:
        graph_filename = filename[: filename.index(".wrl")] + ".mrg"
    else:
        graph_filename = filename

    with open(graph_filename, "r") as f:
        lines = f.read().split("\n")

    pos = [0]

    def next_line():
        if pos[0] >= len(lines):
            raise MrgFormatError("Unexpected end of file in " + graph_filename)
        line = lines[pos[0]]
        pos[0] += 1
        return line

    def expect(token):
        line = next_line()
        if line.strip().lower() != token.lower():
            raise MrgFormatError(
                "The input file " + graph_filename + " contains wrong input at line "
                + str(pos[0]) + "! Program terminating....."
            )

    expect("attributes{")

    att_num = int(next_line().strip())
    attributes = []
    for _ in range(att_num):
        parts = next_line().split()
        st1, st2 = parts[0], parts[1]

        if st1.lower() == "nan":
            st1 = "0.0"
        if st2.lower() == "nan":
            st2 = "0.0"

        attributes.append(AttributeElement(a=float(st1), l=float(st2)))

    expect("}")

    mrg_num = int(next_line().strip())
    mrg = []

    for _ in range(mrg_num):
        expect("elements{")

        graph_size = int(next_line().strip())

        reeb_graph = []

        for _ in range(graph_size):
            index = int(next_line().strip())

            parts = next_line().split()
            left_b, right_b = float(parts[0]), float(parts[1])

            parts = next_line().split()
            tsets = [int(tok) for tok in parts]

            line = next_line().strip()
            if line.upper() == "NULL":
                children = None
            else:
                children = [int(tok) for tok in line.split()]

            el = RNode()
            el.index = index
            el.left_bound = left_b
            el.right_bound = right_b
            el.Tsets = tsets
            el.children = children
            el.MLIST = [0]

            reeb_graph.append([el])

        expect("}")
        expect("connectivity{")

        graph_size = int(next_line().strip())

        for _ in range(graph_size):
            parts = next_line().split()
            main_ver_ind = int(parts[0])

            adj_ver = reeb_graph[main_ver_ind]

            for tok in parts[1:]:
                ver_to_add = int(tok)
                el_to_add = reeb_graph[ver_to_add][0]
                adj_ver.append(el_to_add)

        expect("}")

        mrg.append(reeb_graph)

    return attributes, mrg

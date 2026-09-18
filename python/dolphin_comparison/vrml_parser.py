"""Port of the VRML parser embedded in src/ExtractReebGraph.java
(main_one's "VRML parser BEGIN/END" section).

WARNING (matches the original). This parser can only handle
specially-formatted face-vertex meshes:

1. 3D points are assumed to be enclosed by strings "point [\\n" and "]\\n".
   Coordinates for each point are specified on a separate line in the file,
   e.g.: "1.5 3.2 0.2,\\n"

2. Face lists are assumed to be enclosed by strings "coordIndex [\\n" and
   "]\\n". Each face must appear on a separate line, e.g.:
   "0, 2, 1, -1,\\n" encodes a triangle face (quad faces are also allowed).

3. Please refer to the sample VRML models for examples of the required
   format.

This is a very literal, offset-arithmetic port of the original -- it
relies on exact index positions of spaces/commas, exactly like the Java
version, rather than a "proper" tokenizer, so that it parses the sample
models identically.
"""

import math

from .point import Point
from .triangle import Triangle


class _EndOfFile(Exception):
    """Raised in place of Java's NullPointerException, which the original
    parser relies on (uncaught) to detect EOF -- see the try/except
    NullPointerException at the end of ExtractReebGraph.main_one."""


def _read_line(f):
    line = f.readline()
    if line == "":
        return None
    return line.rstrip("\r\n")


def _nn(s):
    """Raises _EndOfFile if s is None -- mirrors calling a String method on
    a null reference in Java (NullPointerException)."""
    if s is None:
        raise _EndOfFile()
    return s


def parse_vrml(filename):
    """Parses a (pseudo-)VRML file and returns (points, triangles), where
    points is a list[Point] and triangles is a list[Triangle]."""
    points = []
    triangles = []

    with open(filename, "r") as f:
        try:
            _parse(f, points, triangles)
        except _EndOfFile:
            pass

    return points, triangles


def _parse(f, points, triangles):
    prevsize = 0
    stop2 = False

    while True:
        s = _read_line(f)
        _nn(s)

        if "point" in s and "[" in s and s.rfind("]") < s.rfind("["):
            prevsize = len(points)
            stop2 = False
            _parse_point_block(f, points)

        elif "coordIndex" in s and "[" in s and s.rfind("]") < s.rfind("[") and not stop2:
            stop2 = True
            _parse_coord_index_block(f, points, triangles, prevsize)


def _parse_point_block(f, points):
    stop = False
    while not stop:
        s = _read_line(f)
        _nn(s)
        s = s.strip()

        temp_int = s.find("]")

        if "#IGNORE" in s or temp_int == 0:
            stop = True
        else:
            m = s.find(" ")
            n = s.find(" ", m + 1)
            o = s.find(",")

            temp_int2 = s.find("]")
            temp_int3 = s.find(" ", n + 1)
            if o == -1:
                if temp_int2 == -1:
                    o = len(s)
                else:
                    if temp_int3 == -1:
                        o = temp_int2
                    else:
                        o = temp_int3

            if m != -1:
                p = Point()
                p.X = float(s[0:m])
                p.Y = float(s[m:n])
                p.Z = float(s[n:o])
                points.append(p)

            if "]" in s:
                stop = True


def _parse_coord_index_block(f, points, triangles, prevsize):
    stop = False
    while not stop:
        s = _read_line(f)
        _nn(s)
        s = s.strip()

        temp_int = s.find("]")

        if temp_int == 0:
            stop = True
        else:
            q = s.find("-")
            if q == -1:
                m = n = o = p = -1
            else:
                temp = s[0:q]
                m = temp.find(",", 0)
                n = temp.find(",", m + 1)
                o = temp.find(",", n + 1)
                p = temp.find(",", o + 1)

            if "]" in s:
                stop = True

            if m != -1 and p == -1:
                # triangle face
                a = int(s[0:m])
                b = int(s[m + 2 : n])
                c = int(s[n + 2 : o])

                t = Triangle()
                t.a = prevsize + a
                t.b = prevsize + b
                t.c = prevsize + c

                triangles.append(t)

            elif m != -1 and p != -1:
                # quad face -- split into 2 triangles along the "better"
                # diagonal, chosen from the angles at vertex A
                a = int(s[0:m])
                b = int(s[m + 2 : n])
                c = int(s[n + 2 : o])
                d = int(s[o + 2 : p])

                sa_pt = points[prevsize + a]
                sb_pt = points[prevsize + b]
                sc_pt = points[prevsize + c]
                sd_pt = points[prevsize + d]

                v1 = Point(sb_pt.X - sa_pt.X, sb_pt.Y - sa_pt.Y, sb_pt.Z - sa_pt.Z)
                v2 = Point(sc_pt.X - sa_pt.X, sc_pt.Y - sa_pt.Y, sc_pt.Z - sa_pt.Z)
                v3 = Point(sd_pt.X - sa_pt.X, sd_pt.Y - sa_pt.Y, sd_pt.Z - sa_pt.Z)

                length1 = math.sqrt(v1.X * v1.X + v1.Y * v1.Y + v1.Z * v1.Z)
                length2 = math.sqrt(v2.X * v2.X + v2.Y * v2.Y + v2.Z * v2.Z)
                length3 = math.sqrt(v3.X * v3.X + v3.Y * v3.Y + v3.Z * v3.Z)

                angle1 = math.acos((v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z) / (length1 * length2))
                angle2 = math.acos((v1.X * v3.X + v1.Y * v3.Y + v1.Z * v3.Z) / (length1 * length3))
                angle3 = math.acos((v2.X * v3.X + v2.Y * v3.Y + v2.Z * v3.Z) / (length2 * length3))

                if angle1 > angle2:
                    if angle3 > angle1:
                        sa, sc, sb = prevsize + c, prevsize + d, prevsize + b
                    else:
                        sa, sc, sb = prevsize + b, prevsize + c, prevsize + d
                else:
                    if angle2 > angle3:
                        sa, sc, sb = prevsize + b, prevsize + d, prevsize + c
                    else:
                        sa, sc, sb = prevsize + c, prevsize + d, prevsize + b
                sd = prevsize + a

                t1 = Triangle(sa, sb, sc)
                triangles.append(t1)

                t2 = Triangle(sa, sd, sc)
                triangles.append(t2)

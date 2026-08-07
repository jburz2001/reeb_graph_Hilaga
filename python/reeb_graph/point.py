"""Port of src/Point.java -- properties of a 3D point."""

import math


class Point:
    """Contains properties of a 3D point."""

    __slots__ = ("X", "Y", "Z", "TriangleLocation")

    #: matches Point.epsilon2 in the Java source
    epsilon2 = 0.000001

    def __init__(self, X=0.0, Y=0.0, Z=0.0):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.TriangleLocation = 0  # number of the triangle the point is in

    def is_equal_to(self, P):
        if abs(self.X - P.X) < Point.epsilon2:
            if abs(self.Y - P.Y) < Point.epsilon2:
                if abs(self.Z - P.Z) < Point.epsilon2:
                    return True
        return False

    def get_distance(self, P):
        """Returns the distance between 2 points in 3D space."""
        return math.sqrt((self.X - P.X) ** 2 + (self.Y - P.Y) ** 2 + (self.Z - P.Z) ** 2)

    def is_between(self, A, B):
        return (
            ((self.X <= A.X and self.X >= B.X) or (self.X >= A.X and self.X <= B.X))
            and ((self.Y <= A.Y and self.Y >= B.Y) or (self.Y >= A.Y and self.Y <= B.Y))
            and ((self.Z <= A.Z and self.Z >= B.Z) or (self.Z >= A.Z and self.Z <= B.Z))
            and not self.is_equal_to(A)
            and not self.is_equal_to(B)
        )

    def is_between2(self, A, B):
        """Returns True if this point is in between A and B with a tolerance of epsilon2."""
        if A.X > B.X:
            GX, LX = A.X, B.X
        else:
            GX, LX = B.X, A.X

        if A.Y > B.Y:
            GY, LY = A.Y, B.Y
        else:
            GY, LY = B.Y, A.Y

        if A.Z > B.Z:
            GZ, LZ = A.Z, B.Z
        else:
            GZ, LZ = B.Z, A.Z

        return (
            ((self.X > LX or abs(self.X - LX) < Point.epsilon2) and (self.X < GX or abs(self.X - GX) < Point.epsilon2))
            and ((self.Y > LY or abs(self.Y - LY) < Point.epsilon2) and (self.Y < GY or abs(self.Y - GY) < Point.epsilon2))
            and ((self.Z > LZ or abs(self.Z - LZ) < Point.epsilon2) and (self.Z < GZ or abs(self.Z - GZ) < Point.epsilon2))
        )

    def dot_product(self, A, B, which):
        """Computes the dot product between the coplanar normal to a side of a
        triangle (AB) and vector AP in the plane of the triangle.

        Assumes that A and B are vertices of some triangle, and this point is
        in the plane of that triangle.
        """
        if which == 1:
            return (A.Z - B.Z) * (self.Y - A.Y) + (B.Y - A.Y) * (self.Z - A.Z)
        elif which == 2:
            return (A.Z - B.Z) * (self.X - A.X) + (B.X - A.X) * (self.Z - A.Z)
        else:
            return (A.Y - B.Y) * (self.X - A.X) + (B.X - A.X) * (self.Y - A.Y)

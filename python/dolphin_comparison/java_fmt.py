"""Helper to format a float the way Java's ``Double.toString`` / string
concatenation would, used only so that CompareReebGraph's output log file
name (``log_<pts>_<mu_coeff>_<mrg_size>_<sim_weight>``) matches the
original Java program's naming convention (e.g. ``log_4000_5.0E-4_128_0.5``
from the README), for anyone relying on that filename pattern.

This reformats Python's own shortest round-trip digits (from ``repr``) into
Java's layout rules; it is not a from-scratch reimplementation of Java's
grisu-style algorithm, but produces identical output for ordinary CLI
arguments (simple decimals) such as ``mu_coeff`` and ``sim_weight``.
"""

import math
from decimal import Decimal


def java_double_str(x):
    x = float(x)

    if x != x:
        return "NaN"
    if x == float("inf"):
        return "Infinity"
    if x == float("-inf"):
        return "-Infinity"
    if x == 0.0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"

    sign = "-" if x < 0 else ""
    x_abs = abs(x)

    _, digits, exponent = Decimal(repr(x_abs)).as_tuple()
    digits_str = "".join(str(d) for d in digits)
    num_digits = len(digits_str)

    # scientific exponent e such that x_abs == d1.d2d3... * 10**e
    sci_exp = exponent + num_digits - 1

    if -3 <= sci_exp < 7:
        if sci_exp >= 0:
            int_part_len = sci_exp + 1
            if num_digits <= int_part_len:
                int_part = digits_str + "0" * (int_part_len - num_digits)
                frac_part = "0"
            else:
                int_part = digits_str[:int_part_len]
                frac_part = digits_str[int_part_len:]
        else:
            int_part = "0"
            frac_part = "0" * (-sci_exp - 1) + digits_str
        return sign + int_part + "." + frac_part
    else:
        first = digits_str[0]
        rest = digits_str[1:] if len(digits_str) > 1 else "0"
        if rest == "":
            rest = "0"
        return sign + first + "." + rest + "E" + str(sci_exp)

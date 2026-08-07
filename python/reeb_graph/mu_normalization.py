"""Port of src/MuNormalization.java -- normalization of mu values."""


class MuNormalization:
    def normalize(self, mu_values):
        # looks for min and max
        minimum = mu_values[0]
        maximum = mu_values[0]
        for i in range(len(mu_values)):
            temp = mu_values[i]

            if temp > maximum:
                maximum = temp

            if temp < minimum:
                minimum = temp

        # Normalize values using formula
        # MUn(V) = (MU(V) - minimum) / maximum
        for i in range(len(mu_values)):
            temp = mu_values[i]
            mu_values[i] = (temp - minimum) / maximum

        return mu_values

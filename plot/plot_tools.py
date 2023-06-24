__all__ = ["MultipleTicks"]

import numpy as np
import matplotlib.pyplot as plt


class MultipleTicks():
    """Class for formatting multiples of some fractional value in LaTeX as ticks."""
    # reference: https://stackoverflow.com/a/53586826
    
    def __init__(self, denominator=1, number=np.pi, latex="\pi", number_in_frac=True, fracformat=r"\frac{%s}{%s}"):
        """_summary_

        Args:
            denominator (float, optional): Number of ticks between integer
                multiples of `number`. Defaults to 1.
            number (float, optional): Numeric value of `latex`. Defaults to
                np.pi.
            latex (str, optional): LaTeX string of `number`. Defaults to "\pi".
            number_in_frac (bool, optional): Flag whether `latex` string should
                included in numerator of fraction or outside. Defaults to True.
            fracformat (str, optional): LaTeX format for fraction with first %s
                replaced by numerator and second %s by denominator. Defaults to
                r"\frac{%s}{%s}".
        """        
        self.denominator = denominator
        self.number = number
        self.latex = latex
        self.number_in_frac = number_in_frac
        self.fracformat = fracformat
    
    def scalar_formatter(self, scalar):
        """Format scalar value."""
        if scalar == 0:
            return "$0$"
        if scalar == 1:
            return "${}$".format(self.latex)
        elif scalar == -1:
            return "$-{}$".format(self.latex)
        else:
            return "${}{}$".format(scalar, self.latex)
    
    def fraction_formatter(self, num, den):
        """Format fractional value."""
        if self.number_in_frac:
            if num == 1:
                return "${}$".format(self.fracformat % (self.latex, den))
            elif num == -1:
                return "$-{}$".format(self.fracformat % (self.latex, den))
            elif num < -1:
                return "$-{}$".format(self.fracformat % (str(-num) + self.latex, den))
            else:
                return "${}$".format(self.fracformat % (str(num) + self.latex, den))
        else:
            if num < 0:
                return "$-{}{}$".format(self.fracformat % (-num, den), self.latex)
            else:
                return "${}{}$".format(self.fracformat % (num, den), self.latex)
    
    def multiple_formatter(self, x, pos):
        """Format value as scalar or fraction."""
        if self.denominator <= 1:
            scalar = int(np.rint(x / self.number))
            return self.scalar_formatter(scalar)
        else:
            # cancel gcd
            den = self.denominator
            num = int(np.rint(x * den / self.number))
            gcd = np.gcd(num, den)
            num, den = int(num / gcd), int(den / gcd)
            # format fraction
            if den == 1:
                return self.scalar_formatter(num)
            else:
                return self.fraction_formatter(num, den)

    def locator(self):
        """Return matplotlib locator."""
        if self.denominator <= 1:
            scalar = int(np.rint(1 / self.denominator))
            return plt.MultipleLocator(scalar * self.number)
        else:
            return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        """Return matplotlib formatter."""
        return plt.FuncFormatter(self.multiple_formatter)

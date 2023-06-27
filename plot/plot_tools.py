__all__ = ["MultipleTicks"]

import numpy as np
import matplotlib.pyplot as plt


class MultipleTicks():
    """Locate and format ticks at multiples of some constant in LaTeX.

    This class complements matplotlib.ticker.MultipleLocator [1] by adding the
    possibility to format ticks as fractions and/or to represent the constant
    with a non-numeric symbol.
    
    The formatting is inspired from [2].

    Examples:
        Format ticks as multiples of pi/2:
        ```
        ticks = MultipleTicks((np.pi, r"\pi"), 2)
        axis.xaxis.set_major_locator(ticks.locator())
        axis.xaxis.set_major_formatter(ticks.formatter())
        ```

        Format ticks as multiples of 3/2:
        ```
        ticks = MultipleTicks(3, 2)
        axis.xaxis.set_major_locator(ticks.locator())
        axis.xaxis.set_major_formatter(ticks.formatter())
        ```

    References:
        [1] https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.MultipleLocator
        [2] https://stackoverflow.com/a/53586826
    """
    
    def __init__(self, num, den=1, number_in_frac=False, fracformat=r"\frac{%s}{%s}"):
        """Public constructor.

        Args:
            num (float or (float, str)): The value of the constant. If the
                constant should be represented with a non-numeric symbol, a
                tuple (num_val, num_latex) must be provided.
            den (float, optional): The denominator value for ticks at multiples
                of fractions. It must be at least 1. Defaults to 1.
            number_in_frac (bool, optional): Flag whether the non-numeric symbol
                should be in or outside of the fraction. This parameter is
                ignored if num is of type float. Defaults to False.
            fracformat (str, optional): LaTeX format for fraction with first %s
                replaced by numerator and second %s by denominator. Defaults to
                r"\frac{%s}{%s}".
        """
        if isinstance(num, tuple):
            self.num_val, self.num_latex = num
        else:
            self.num_val, self.num_latex = num, None
        self.den = den
        self.symbol_in_frac = number_in_frac
        self.fracformat = fracformat
    
    def _format_scalar(self, scalar):
        """Format scalar value."""
        if self.num_latex is None:
            # format with numeric symbol
            scalar = scalar * self.num_val
            return "${}$".format(scalar)
        else:
            # format with non-numeric symbol
            if scalar == 0:
                return "$0$"
            if scalar == 1:
                return "${}$".format(self.num_latex)
            elif scalar == -1:
                return "$-{}$".format(self.num_latex)
            else:
                return "${}{}$".format(scalar, self.num_latex)
    
    def _format_fraction(self, num, den):
        """Format fractional value."""
        if self.num_latex is None:
            # format with numeric symbol
            num = num * self.num_val
            if num >= 1:
                return "${}$".format(self.fracformat % (num, den))
            else: # num <= -1
                 return "$-{}$".format(self.fracformat % (-num, den))
        else:
            # format with non-numeric symbol
            if self.symbol_in_frac:
                if num == 1:
                    return "${}$".format(self.fracformat % (self.num_latex, den))
                elif num == -1:
                    return "$-{}$".format(self.fracformat % (self.num_latex, den))
                elif num > 1:
                    return "${}$".format(self.fracformat % (str(num) + self.num_latex, den))
                else: # num < -1
                    return "$-{}$".format(self.fracformat % (str(-num) + self.num_latex, den))
            else:
                if num >= 1:
                    return "${}{}$".format(self.fracformat % (num, den), self.num_latex)
                else: # num <= -1
                    return "$-{}{}$".format(self.fracformat % (-num, den), self.num_latex)
    
    def _format_multiple(self, x, pos):
        """Format value as scalar or fraction."""
        if self.den <= 1:
            # format scalar
            scalar = int(np.rint(x / self.num_val))
            return self._format_scalar(scalar)
        else:
            # cancel gcd
            den = self.den
            num = int(np.rint(x * den / self.num_val))
            gcd = np.gcd(num, den)
            num, den = int(num / gcd), int(den / gcd)
            # format fraction
            if den == 1:
                return self._format_scalar(num)
            else:
                return self._format_fraction(num, den)

    def locator(self):
        """Return Matplotlib locator."""
        if self.den <= 1:
            scalar = int(np.rint(1 / self.den))
            return plt.MultipleLocator(scalar * self.num_val)
        else:
            return plt.MultipleLocator(self.num_val / self.den)

    def formatter(self):
        """Return Matplotlib formatter."""
        return plt.FuncFormatter(self._format_multiple)

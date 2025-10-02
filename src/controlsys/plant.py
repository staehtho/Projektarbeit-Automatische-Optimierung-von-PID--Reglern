import numpy as np


class Plant:
    """
    Represents a linear time-invariant (LTI) system defined by its transfer function.

    Parameters
    ----------
    num : list[float] | np.ndarray
        Coefficients of the numerator of the transfer function, in descending powers of s.
    dec : list[float] | np.ndarray
        Coefficients of the denominator of the transfer function, in descending powers of s.
    """

    def __init__(self,
                 num: list[float] | np.ndarray,
                 dec: list[float] | np.ndarray
                 ) -> None:

        self._num = np.array(num, copy=False, dtype=float)
        self._dec = np.array(dec, copy=False, dtype=float)

        self._t1 = 0

    # ******************************
    # Attributes
    # ******************************

    @property
    def num(self) -> np.ndarray:
        """Numerator coefficients of the transfer function."""
        return self._num

    @property
    def dec(self) -> np.ndarray:
        """Denominator coefficients of the transfer function."""
        return self._dec

    @property
    def t1(self) -> float:
        return self._t1

    # ******************************
    # Methods
    # ******************************

    def system(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Evaluate the transfer function of the plant at a given complex frequency s.

        Parameters
        ----------
        s : complex or np.ndarray
            The Laplace variable (can be scalar or array of complex numbers).

        Returns
        -------
        complex or np.ndarray
            Value of the transfer function at s.
            If s is an array, returns an array of the same shape.

        Notes
        -----
        Index 0 of _num and _dec corresponds to s^0. The coefficients are reversed
        internally to match numpy.polyval, which expects the highest power first.
        """
        return np.polyval(self._num[::-1], s) / np.polyval(self._dec[::-1], s)


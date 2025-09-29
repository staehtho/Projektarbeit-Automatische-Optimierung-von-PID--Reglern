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

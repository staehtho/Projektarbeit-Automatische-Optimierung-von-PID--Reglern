import numpy as np


class Plant:
    """
    Represents a linear time-invariant (LTI) system defined by its transfer function.

    Args:
        num (list[float] | np.ndarray): Numerator coefficients of the transfer function,
            in descending powers of s.
        dec (list[float] | np.ndarray): Denominator coefficients of the transfer function,
            in descending powers of s.

    Examples:
        >>> plant = Plant(num=[1], dec=[1, 2, 1])
        >>> plant.system(1j)
    """

    def __init__(
        self,
        num: list[float] | np.ndarray,
        dec: list[float] | np.ndarray
    ) -> None:
        self._num = np.array(num, copy=False, dtype=float)
        self._dec = np.array(dec, copy=False, dtype=float)
        self._t1 = 1  # dominant time constant for derivative filter calculation

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
        """Dominant time constant (used for derivative filter)."""
        return self._t1

    # ******************************
    # Methods
    # ******************************

    def system(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Evaluate the transfer function of the plant at a given complex frequency `s`.

        Args:
            s (complex | np.ndarray): Laplace variable (σ + jω). Can be scalar or array.

        Returns:
            complex | np.ndarray: Value of the transfer function at `s`. Returns an array
            of the same shape if `s` is an array.

        Notes:
            - Index 0 of `_num` and `_dec` corresponds to s**0.
            - Coefficient arrays are reversed internally to match `numpy.polyval`, which
              expects the highest power first.

        Examples:
            >>> plant = Plant(num=[1], dec=[1, 2, 1])
            >>> plant.system(1j)
        """
        return np.polyval(self._num, s) / np.polyval(self._dec, s)

    def __format__(self, format_spec: str) -> str:
        """
        Format the transfer function as a string.

        Args:
            format_spec (str): Format type. Supported:
                - "mat": MATLAB-style string representation.

        Returns:
            str: Transfer function formatted as a string.

        Raises:
            NotImplementedError: If the format_spec is not supported.

        Examples:
            >>> plant = Plant(num=[1], dec=[1, 2, 1])
            >>> format(plant, "mat")
        """
        format_spec = format_spec.replace(" ", "")
        if format_spec == "mat":
            sys_str = "("
            sys_str += "+".join([f"{self._num[i]} * s ^{self._num.shape[0] - 1 - i}"
                                 for i in range(self._num.shape[0])])
            sys_str += ") / ("
            sys_str += "+".join([f"{self._dec[i]} * s ^{self._dec.shape[0] - 1 - i}"
                                 for i in range(self._dec.shape[0])])
            sys_str += ")"
        else:
            raise NotImplementedError
        return sys_str

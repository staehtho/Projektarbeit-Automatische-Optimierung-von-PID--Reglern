import numpy as np
from scipy.signal import tf2ss
from scipy.integrate import solve_ivp
from typing import Any, Callable


class Plant:
    """
    Represents a linear time-invariant (LTI) system defined by its transfer function.

    Args:
        num (list[float] | np.ndarray): Numerator coefficients of the transfer function,
            in descending powers of s.
        den (list[float] | np.ndarray): Denominator coefficients of the transfer function,
            in descending powers of s.

    Examples:
        >>> plant = Plant(num=[1],den=[1, 2, 1])
        >>> plant.system(1j)
    """

    def __init__(
            self,
            num: list[float] | np.ndarray,
            den: list[float] | np.ndarray
    ) -> None:
        self._num = np.array(num, copy=False, dtype=float)
        self._den = np.array(den, copy=False, dtype=float)
        self._t1 = 1  # dominant time constant for derivative filter calculation

        # Transfer function to state-space representation.
        self._A: np.ndarray
        self._B: np.ndarray
        self._C: np.ndarray
        self._D: np.ndarray

        self._A, self._B, self._C, self._D = tf2ss(self._num, self._den)

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
            >>> plant = Plant(num=[1],den=[1, 2, 1])
            >>> format(plant, "plant")
            'tf([1], [1 2 1])'
        """
        # Whitespace entfernen und lower() für Sicherheit
        format_spec = format_spec.strip().lower()

        if format_spec == "plant":
            num_str = "[" + " ".join(map(str, self._num)) + "]"
            den_str = "[" + " ".join(map(str, self._den)) + "]"
            return f"tf({num_str}, {den_str})"
        else:
            raise NotImplementedError(f"Unsupported format specifier: '{format_spec}'")

    # ******************************
    # Attributes
    # ******************************

    @property
    def num(self) -> np.ndarray:
        """Numerator coefficients of the transfer function."""
        return self._num

    @property
    def den(self) -> np.ndarray:
        """Denominator coefficients of the transfer function."""
        return self._den

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
            - Index 0 of `_num` and `_den` corresponds to s**0.
            - Coefficient arrays are reversed internally to match `numpy.polyval`, which
              expects the highest power first.

        Examples:
            >>> plant = Plant(num=[1],den=[1, 2, 1])
            >>> plant.system(1j)
        """
        return np.polyval(self._num, s) / np.polyval(self._den, s)

    def step_response(self, t0: float = 0, t1: float = 10, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Compute the step response of the plant.

        Args:
            t0 (float): Start time of the simulation. Defaults to 0.
            t1 (float): End time of the simulation. Defaults to 10.
            dt (float): Time step for simulation. Defaults to 0.01.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - t_eval (np.ndarray): Array of time points.
                - y_hist (np.ndarray): Step response of the system at each time point.

        Example:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> t, y = plant.step_response()
        """
        u = lambda t: 1
        return self.system_response(u, t0, t1, dt)

    def system_response(self,
                        u: Callable[[float], float],
                        t0: float = 0,
                        t1: float = 10,
                        dt: float = 0.01,
                        x0: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the plant response to a given input signal.

        Args:
            u (Callable[[float], float]): Input function of time, u(t).
            t0 (float): Start time of the simulation. Defaults to 0.
            t1 (float): End time of the simulation. Defaults to 10.
            dt (float): Time step for simulation. Defaults to 0.01.
            x0 (np.ndarray | None): Initial state of the system. If None, initialized to zero.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - t_eval (np.ndarray): Array of time points.
                - y_hist (np.ndarray): System output at each time point.

        Example:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> u = lambda t: np.sin(t)
            >>> t, y = plant.system_response(u, t0=0, t1=5, dt=0.01)
        """
        if x0 is None:
            x0 = np.zeros(self._A.shape[0])

        x = x0
        t_eval = np.arange(t0, t1, dt)
        y_hist = []

        for t in t_eval:
            x, y = self._tf2ivp(u(t), t, t + dt, x)
            y_hist.append(y)

        return t_eval, np.array(y_hist)

    def tf2ivp(self, u: float,
               t0: float,
               t1: float,
               x0: np.ndarray,
               method: str = "RK23"
               ) -> tuple[np.ndarray, np.ndarray]:
        """Perform a single integration step of the plant's state-space system.

        Args:
            u (float): Input value held constant over the integration step.
            t0 (float): Start time of the integration step.
            t1 (float): End time of the integration step.
            x0 (np.ndarray): Initial state at time t0.
            method (str): Integration method to use (passed to solve_ivp). Defaults to 'RK23'.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - x_next (np.ndarray): System state at t1.
                - y (float): System output at t1.

        Example:
            >>> x_next, y = plant.tf2ivp(u=1.0, t0=0, t1=0.01, x0=np.zeros(2))
        """

        def dx_dt(t: float, x: np.ndarray) -> np.ndarray:
            return (self._A @ x + self._B.flatten() * u).flatten()

        sol: Any = solve_ivp(dx_dt, (t0, t1), x0, t_eval=[t1], method=method, max_step=0.1)
        y = (self._C @ sol.y[:, -1] + self._D * u).item()
        return sol.y[:, -1], y


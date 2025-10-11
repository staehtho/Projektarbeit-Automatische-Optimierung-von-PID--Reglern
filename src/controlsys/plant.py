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
        # TODO: T1 berechnen
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

    def get_system_order(self) -> int:
        return self._A.shape[0]

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

    def step_response(
            self,
            t0: float = 0,
            t1: float = 10,
            dt: float = 0.01,
            method: str = "RK23"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the time-domain step response of the LTI plant.

        Simulates the output y(t) of the system for a unit step input `u(t) = 1`
        using either a fixed-step RK4 method or SciPy’s adaptive `solve_ivp` integrator.

        Args:
            t0 (float, optional): Start time of the simulation [s]. Defaults to 0.
            t1 (float, optional): End time of the simulation [s]. Defaults to 10.
            dt (float, optional): Simulation time step [s]. Defaults to 0.01.
            method (str, optional): Integration method.
                - `"RK4"`: fixed-step fourth-order Runge–Kutta.
                - any other value: uses SciPy’s adaptive `solve_ivp`.
                Defaults to `"RK23"`.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **t_eval** (`np.ndarray`): Time vector.
                - **y_hist** (`np.ndarray`): Output response y(t) for each time point.

        Example:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> t, y = plant.step_response(t1=5, dt=0.01, method="RK4")
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(t, y)
            >>> plt.xlabel("Time [s]")
            >>> plt.ylabel("Output y(t)")
            >>> plt.grid(True)
            >>> plt.show()
        """
        u = lambda t: 1
        return self.system_response(u, t0, t1, dt, method=method)

    def system_response(self,
                        u: Callable[[float], float],
                        t0: float = 0,
                        t1: float = 10,
                        dt: float = 0.01,
                        x0: np.ndarray | None = None,
                        method: str = "RK23"
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the time-domain response of the plant to a given input signal.

        This function integrates the continuous-time state-space system using either
        a fixed-step RK4 integrator or SciPy’s adaptive `solve_ivp` method. The input
        signal `u(t)` is evaluated at each time step.

        Args:
            u (Callable[[float], float]): Input function of time, u(t).
            t0 (float, optional): Simulation start time [s]. Defaults to 0.
            t1 (float, optional): Simulation end time [s]. Defaults to 10.
            dt (float, optional): Fixed simulation time step [s]. Defaults to 0.01.
            x0 (np.ndarray | None, optional): Initial system state. If None, zeros are used.
            method (str, optional): Integration method.
                - "RK4": fixed-step fourth-order Runge–Kutta method.
                - otherwise: uses SciPy’s `solve_ivp` (e.g., "RK23" or "RK45").
                Defaults to "RK23".

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **t_eval** (`np.ndarray`): Time vector.
                - **y_hist** (`np.ndarray`): System output y(t) for each time step.

        Example:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> u = lambda t: np.sin(t)
            >>> t, y = plant.system_response(u, t0=0, t1=5, dt=0.01, method="RK4")
        """

        if x0 is None:
            x0 = np.zeros(self._A.shape[0])

        x = x0
        t_eval = np.arange(t0, t1, dt)
        y_hist = []

        for t in t_eval:
            if method == "RK4":
                x, y = self.rk4_step(u(t), dt, x)
            else:
                x, y = self.tf2ivp(u(t), t, t + dt, x)
            y_hist.append(y)

        return t_eval, np.array(y_hist)

    def tf2ivp(self, u: float,
               t0: float,
               t1: float,
               x0: np.ndarray,
               method: str = "RK23"
               ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the LTI state-space system over one time interval.

        Uses SciPy’s `solve_ivp` to integrate from `t0` to `t1` with constant input `u`.

        Args:
            u (float): Constant input value over [t0, t1].
            t0 (float): Start time of integration [s].
            t1 (float): End time of integration [s].
            x0 (np.ndarray): System state at t0.
            method (str, optional): Integration method passed to `solve_ivp`.
                Defaults to "RK23".

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - **x_next** (`np.ndarray`): State vector at time t1.
                - **y** (`float`): System output y(t1).

        Example:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> x_next, y = plant.tf2ivp(u=1.0, t0=0, t1=0.01, x0=np.zeros(2))
        """

        def dx_dt(t: float, x: np.ndarray) -> np.ndarray:
            return (self._A @ x + self._B.flatten() * u).flatten()

        sol: Any = solve_ivp(dx_dt, (t0, t1), x0, t_eval=[t1], method=method, max_step=0.1)
        y = (self._C @ sol.y[:, -1] + self._D * u).item()
        return sol.y[:, -1], y

    def rk4_step(self, u: float, dt: float, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Perform a single fixed-step RK4 integration for the LTI plant.

        Computes the state update for `dx/dt = A·x + B·u` over one time step `dt`
        assuming `u` is constant during the interval. This method is typically used
        in fixed-step control simulations (e.g. PID loops).

        Args:
            u (float): Constant input during the integration step.
            dt (float): Time step [s].
            x (np.ndarray): Current state vector.

        Returns:
            tuple[np.ndarray, float]:
                - **x_next** (`np.ndarray`): Updated state vector at t + dt.
                - **y** (`float`): Corresponding system output y(t + dt).

        Example:
            >>> plant = Plant(num=[1], den=[1, 2, 1])
            >>> x_next, y = plant.rk4_step(u=1.0, dt=0.01, x=np.zeros(2))
        """

        def dx_dt(x):
            return (self._A @ x + self._B.flatten() * u).flatten()

        k1 = dx_dt(x)
        k2 = dx_dt(x + 0.5 * dt * k1)
        k3 = dx_dt(x + 0.5 * dt * k2)
        k4 = dx_dt(x + dt * k3)

        x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = (self._C @ x_next + self._D * u).item()

        return x_next, y



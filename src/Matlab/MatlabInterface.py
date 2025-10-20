import matlab.engine
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
import inspect
# ToDo: In README einfügen: Umgebungsvariable einfügen


class MatlabInterface:
    """Interface to MATLAB and Simulink for simulations and Bode plot analysis."""

    def __init__(self):
        self.engine = None
        self._t: np.ndarray = np.array([])
        self._values: dict[str, dict[str, str | np.ndarray]] = {}

    def __enter__(self):
        """Start MATLAB engine in no-JVM, no-display mode."""
        print("Matlab starting...")
        self.engine = matlab.engine.start_matlab("-nojvm -nodisplay")
        print("Matlab is running...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Quit MATLAB engine on exit."""
        print("Matlab closing...")
        if self.engine is not None:
            self.engine.quit()

    def run_simulation(
            self,
            simulink_model: str,
            variable_name_out: str,
            start_time: float = 0,
            stop_time: float = 10,
            solver: str = "ode45",
            max_step: float | str = 'auto',
            min_step: float | str = 'auto',
            abs_tol: float = 1e-4,
            rel_tol: float = 1e-4,
            **kwargs: Any
    ) -> None:
        """
        Run a Simulink simulation using the MATLAB Engine for Python and
        retrieve the specified output signals.

        The method automatically sets MATLAB's working directory to the directory of
        the Python script calling this method. The Simulink model file must be
        accessible from that directory.

        Args:
            simulink_model (str):
                Name of the Simulink model to simulate (with or without the `.slx` extension).
                The model file must be located in the same directory as the calling Python script.

            variable_name_out (str):
                Name of the variable defined by a 'To Workspace' block in the Simulink model
                whose data should be retrieved after the simulation.

            start_time (float, optional):
                Simulation start time in seconds. Defaults to 0.

            stop_time (float, optional):
                Simulation stop time in seconds. Defaults to 10.

            solver (str, optional):
                Name of the solver algorithm to use (e.g., `"ode45"`, `"ode15s"`, `"ode23tb"`).
                Defaults to `"ode45"`.

            max_step (float | str, optional):
                Maximum solver step size. Can be a numeric value or `'auto'`. Defaults to `'auto'`.

            min_step (float | str, optional):
                Minimum solver step size. Can be a numeric value or `'auto'`. Defaults to `'auto'`.

            abs_tol (float, optional):
                Absolute tolerance of the solver. Defaults to `1e-4`.

            rel_tol (float, optional):
                Relative tolerance of the solver. Defaults to `1e-4`.

            **kwargs (Any):
                Additional MATLAB workspace variables to define before running the simulation.
                Each key–value pair is sent to MATLAB via `eng.workspace`.

        Raises:
            FileNotFoundError:
                If the specified Simulink model file is not found in the caller's directory.

            TypeError:
                If one or more variables in `**kwargs` are of a type unsupported by the MATLAB Engine.

            matlab.engine.EngineError:
                If MATLAB encounters an error during model loading or simulation execution.

        Returns:
            None

        Notes:
            - The simulation results are stored internally (e.g., in `self.output_data` or
              another class attribute, depending on the implementation).
            - To access the output data, retrieve it from the instance after the simulation completes.

        Examples:
            >>> # Example: Run a step response simulation
            >>> mat.run_simulation(
            ...     simulink_model='stepresponse',
            ...     variable_name_out='yout',
            ...     start_time=0,
            ...     stop_time=10,
            ...     solver='ode45',
            ...     G="tf([1, 2], [1, 2, 3])"
            ... )
        """

        print("Simulation is running...")

        # Determine directory of the calling Python file dynamically
        frame = inspect.stack()[1]  # [0] = current function, [1] = caller
        caller_file = os.path.abspath(frame.filename)
        path = os.path.dirname(caller_file)

        # Set MATLAB working directory to caller's location
        self.engine.cd(path, nargout=0)

        # Write all provided variables into the MATLAB workspace
        self.write_in_workspace(**kwargs)

        # Run Simulink model
        self.engine.eval(f"""
            simIn = Simulink.SimulationInput('{simulink_model}');
            simIn = simIn.setModelParameter('StartTime', '{start_time}', ...
            'StopTime', '{stop_time}', ...
            'Solver', '{solver}', ...
            'MaxStep', '{max_step}', ...
            'MinStep', '{min_step}', ...
            'AbsTol', '{abs_tol}', ...
            'RelTol', '{rel_tol}');
            out = sim(simIn);
        """, nargout=0)

        # Retrieve time vector
        self._t = np.array(self.engine.eval(f"out.{variable_name_out}.time", nargout=1)).flatten()

        # Retrieve all output signals
        num_signals: int = int(self.engine.eval(f"numel(out.{variable_name_out}.signals)", nargout=1))
        for i in range(1, num_signals + 1):
            value: np.ndarray = np.array(
                self.engine.eval(f"out.{variable_name_out}.signals({i}).values", nargout=1)
            ).flatten()
            label: str = self.engine.eval(f"out.{variable_name_out}.signals({i}).label", nargout=1)
            title: str = self.engine.eval(f"out.{variable_name_out}.signals({i}).title", nargout=1)
            self._values[f"value_{label}"] = {"value": value, "label": label, "title": title}

    def plot_simulation(
        self,
        figure_name: str,
        title: str,
        *,
        path: str = "",
        save: bool = False,
        show: bool = False
    ) -> None:
        """
        Plot the stored simulation results.

        Args:
            figure_name (str): Name of the matplotlib figure and filename if saved.
            title (str): Title of the plot.
            path (str, optional): Directory path to save figure. Defaults to "".
            save (bool, optional): Save figure as PNG. Defaults to False.
            show (bool, optional): Display plot interactively. Defaults to False.

        Examples:
            >>> MatlabInterface.plot_simulation("step_response", "Step Response", path="C:/Plots", save=True)
        """
        plt.figure(figure_name)
        for key, signal in self._values.items():
            plt.plot(self._t, signal["value"], label=signal["label"])
        plt.legend()
        plt.grid()
        plt.title(title)

        if save:
            save_path = os.path.join(path, f"{figure_name}.png") if path else f"{figure_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    @property
    def t(self) -> np.ndarray:
        """Return simulation time vector."""
        return self._t

    @property
    def values(self) -> dict[str, dict[str, str | np.ndarray]]:
        """Return dictionary of stored simulation signals."""
        return self._values

    def write_in_workspace(self, **kwargs: Any) -> None:
        """
        Write Python variables into the MATLAB workspace.

        Args:
            **kwargs: Key-value pairs to write into MATLAB:
                - str: Treated as MATLAB expression (eval).
                - list, tuple, np.ndarray: Converted to matlab.double.
                - int, float: Written as scalar values.

        Raises:
            TypeError: If unsupported type is passed.

        Examples:
            >>> MatlabInterface.write_in_workspace(K=5, G="tf([1],[1 2 1])")
        """
        for key, value in kwargs.items():
            if isinstance(value, str):
                param_str = f"{key} = {value}"
            elif isinstance(value, (list, tuple)):
                param_str = f"{key} = {matlab.double(value)};"
            else:
                param_str = f"{key} = {value};"
            self.engine.eval(param_str, nargout=0)

    def bode(
        self,
        sys: str,
        low_exp: float,
        high_exp: float,
        num_points: int,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Bode plot data for a transfer function defined in MATLAB.

        Args:
            sys (str): Name of the transfer function variable in MATLAB workspace.
            low_exp (float): Lower exponent for logspace (10**low_exp).
            high_exp (float): Upper exponent for logspace (10**high_exp).
            num_points (int): Number of frequency points.
            **kwargs: Additional variables to write into MATLAB workspace.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Magnitude (dB), phase (rad), frequency (rad/s).
        """
        self.write_in_workspace(**kwargs)
        self.engine.eval(f"w = logspace({low_exp},{high_exp},{num_points});", nargout=0)
        mag, phase, wout = self.engine.eval(f"bode({sys}, w)", nargout=3)

        mag: np.ndarray = np.squeeze(np.array(mag))
        mag_db: np.ndarray = 20 * np.log10(mag)
        phase: np.ndarray = np.squeeze(np.array(phase))
        wout: np.ndarray = np.squeeze(np.array(wout))

        return mag_db, phase, wout

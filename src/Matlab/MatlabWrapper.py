import matlab.engine
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


class MatlabWrapper:

    def __init__(self):
        self.engine = None
        self._t: np.ndarray = np.array([])
        self._values: dict[str, dict[str, str | np.ndarray]] = {}

    def __enter__(self):
        print("Matlab starting...")
        self.engine = matlab.engine.start_matlab("-nojvm -nodisplay")
        print("Matlab is running...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
            max_step: float = 0.01,
            **kwargs: Any
    ) -> None:
        """
        Runs a Simulink simulation and stores the output signals in Python.

        Args:
            simulink_model (str): Name of the Simulink model or absolute path to the .slx file.
                If a path is provided, it will be loaded directly.
            variable_name_out (str): Name of the 'To Workspace' variable in Simulink to retrieve.
            start_time (Optional[float]): Simulation start time in seconds. Defaults to 0.
            stop_time (Optional[float]): Simulation stop time in seconds. Defaults to 10.
            solver (Optional[str]): Solver algorithm used by Simulink for numerical integration.
                Common options include "ode45", "ode23", "ode15s", etc.
                Defaults to "ode45".
            max_step (Optional[float]): Maximum step size for the solver in seconds. Controls the resolution
                  of the simulation. Smaller values increase accuracy but may slow
                  down the simulation. Defaults to 0.01.
            kwargs (Any): Arbitrary MATLAB workspace variables to set before simulation.
                   Example: G="tf([2],[1 3 2])" or K=5.

        Side Effects:
            - Sets workspace variables in MATLAB according to kwargs.
            - Executes the simulation in MATLAB.
            - Populates self._t with the simulation time vector.
            - Populates self._values with all signals from the specified To Workspace variable.

        Example:
            - run_simulation('model', 'y', stop_time=20, G='tf([2],[1 3 2])')
            - run_simulation('C:/Users/User/Documents/MATLAB/model.slx', 'y', stop_time=20, G='tf([2],[1 3 2])')
        """

        print("Simulation is running...")

        # Überprüfen ob simulink_model ein absoluter Pfad ist
        if os.path.isabs(simulink_model):
            model_str = simulink_model.replace("\\", "/")  # MATLAB mag '/'
        else:
            model_str = simulink_model

        # Set MATLAB workspace variables
        for key, value in kwargs.items():
            if isinstance(value, str):
                # Direkt als MATLAB Expression übergeben
                param_str = f"{key} = {value}"
            elif isinstance(value, (list, tuple)):
                # Listen in matlab.double umwandeln
                param_str = f"{key} = {matlab.double(value)};"
            else:
                # Zahlen direkt
                param_str = f"{key} = {value};"
            self.engine.eval(param_str, nargout=0)

        # Simulation starten
        self.engine.eval(f"""
            simIn = Simulink.SimulationInput('{model_str}');
            simIn = simIn.setModelParameter('StartTime', '{start_time}', ...
            'StopTime', '{stop_time}', ...
            'Solver', '{solver}', ...
            'MaxStep', '{max_step}');
            out = sim(simIn);
        """, nargout=0)

        # Zeitvektor extrahieren
        self._t = np.array(self.engine.eval(f"out.{variable_name_out}.time", nargout=1)).flatten()

        # Anzahl Signale bestimmen
        num_signals: int = int(self.engine.eval(f"numel(out.{variable_name_out}.signals)", nargout=1))

        # Signale auslesen
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
        Plots the stored simulation results.

        Args:
            figure_name (str): Name of the matplotlib figure.
                Also used as the filename if save=True.
            title (str): Title of the plot.
            path (str, optional): Directory path where the figure should be saved.
                If empty, the current working directory is used. Defaults to "".
            save (bool, optional): If True, saves the plot as a PNG file using
                figure_name (and path if provided). Defaults to False.
            show (bool, optional): If True, displays the plot in a window.
                Defaults to False.

        Side Effects:
            - Creates a matplotlib figure containing all signals stored in self._values.
            - If save=True, a file named '<figure_name>.png' is written
              to `path` (or cwd if empty).
            - If show=True, the plot window will be displayed interactively.

        Example:
            mat.plot_simulation(
                "step_response",
                "Step Response",
                path="C:/Users/Me/Plots",
                save=True,
                show=False
            )
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
        return self._t

    @property
    def values(self) -> dict[str, dict[str, str | np.ndarray]]:
        return self._values

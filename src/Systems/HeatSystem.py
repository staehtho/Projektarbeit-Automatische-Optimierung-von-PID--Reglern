# -*- coding: utf-8 -*-
"""
Created on Thu May 3 10:29:36 2024

@author: lukas
"""

import matplotlib.pyplot as plt
from Systems.closedLoopSystem2 import ClosedLoopSystem2


class HeatSystem(ClosedLoopSystem2):
    """
    Heat transfer system with two interconnected thermal masses (T1 and T2)
    controlled by a PID controller.
    """

    def __init__(self,
                 Kp: float,
                 Ki: float | None = None,
                 Kd: float | None = None,
                 Ti: float | None = None,
                 Td: float | None = None,
                 pos_sat: float | None = None,
                 neg_sat: float | None = None,
                 K: float = 1.0,
                 setPoint: float = 100.0,
                 c1: float = 500.0,
                 c2: float = 4187.0,
                 A: float = 0.062,
                 alpha1: float = 400.0,
                 alpha2: float = 370.0,
                 m1: float = 1.0,
                 m2: float = 1.0,
                 control: bool = True) -> None:
        """Initializes the heat system with thermal parameters."""
        super().__init__(Kp, Ki, Kd, Ti, Td, pos_sat, neg_sat, K, setPoint, control=control)

        # Thermal parameters
        self.c1: float = c1
        self.c2: float = c2
        self.A: float = A
        self.alpha1: float = alpha1
        self.alpha2: float = alpha2
        self.m1: float = m1
        self.m2: float = m2

    def _system_dynamics(self, t: float, y: list[float]) -> list[float]:
        """
        Defines the system dynamics for the heat transfer system.

        Args:
            t (float): Current time.
            y (list[float]): Current state [T1, T2].

        Returns:
            list[float]: Derivatives [dT1/dt, dT2/dt].
        """
        T1, T2 = y

        # Determine control input
        if self.control:
            T = self._pid_controller(t, y)
        else:
            T = self.mySetPoint

        # Thermal dynamics
        dT1_dt = (self.alpha1 * self.A * (T - T1) - self.alpha2 * self.A * (T1 - T2)) / (self.m1 * self.c1)
        dT2_dt = (self.alpha2 * self.A * (T1 - T2)) / (self.m2 * self.c2)

        return [dT1_dt, dT2_dt]

    @staticmethod
    def plotSystemResponseX(X: list[float], pos_sat: float = 500.0, neg_sat: float = 20.0, control: bool = True) -> None:
        """Plots the step response for [Kp, Ti, Td]."""
        system = HeatSystem(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat, control=control)
        itae_krit, timet, step_response = system.response()
        plt.plot(timet, step_response, color='blue', linewidth=2, label='System Response')
        textstr = f'$K_p = {X[0]}$\n$T_i = {X[1]}$\n$T_d = {X[2]}$'
        plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.title('Heat Transfer System')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"ITAE: {itae_krit}")

    @staticmethod
    def plotSystemResponseX2(X: list[float], pos_sat: float = 500.0, neg_sat: float = 20.0, control: bool = True) -> None:
        """Plots the step response for [Kp, Ki, Kd]."""
        system = HeatSystem(Kp=X[0], Ki=X[1], Kd=X[2], pos_sat=pos_sat, neg_sat=neg_sat, control=control)
        itae_krit, timet, step_response = system.response()
        plt.plot(timet, step_response, color='red', linewidth=2, label='System Response')
        plt.title('Heat Transfer System')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"ITAE: {itae_krit}")

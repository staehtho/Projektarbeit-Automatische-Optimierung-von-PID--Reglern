# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:12:52 2024

@author: lukas
"""

import numpy as np
from scipy.integrate import solve_ivp

class ClosedLoopSystem:
    def __init__(self, Kp, Ki=None, Kd=None, Ti=None, Td=None, pos_sat=None, neg_sat=None, K=1.0, setPoint=1.0, tau_d=0.01, control=True):
        self.myKp = Kp
        if Ki is not None:
            self.myKi = Ki
        else:
            if Ti == 0:
                self.myKi = 0
            else:
                self.myKi = Kp / Ti
        if Kd is not None:
            self.myKd = Kd
        else:
            self.myKd = Kp * Td
        
        self.tau_d = tau_d
        self.myPosSat = pos_sat
        self.myNegSat = neg_sat
        self.myIntegralError = 0.0
        self.myLastError = [0.0, 0.0]
        self.mySetPoint = setPoint
        self.prev_d = 0
        self.control = control
        self.filtered_d = 0
        self.tau = 0.01
        # state
        self.order_i=0
    
    def _pid_controller(self, t, y):
        error = self.mySetPoint - y[self.order_i]
        dt = t - self.myLastError[1]
        if dt > 0:
            error_diff = error - self.myLastError[0]
            derivative = error_diff / dt
            # Apply the filter to the derivative
            alpha = dt / (self.tau + dt)
            self.filtered_d = alpha * derivative + (1 - alpha) * self.filtered_d
            self.myIntegralError += error * dt
        else:
            self.filtered_d = self.filtered_d

        P = self.myKp * error
        I = self.myKi * self.myIntegralError

        if self.myNegSat is not None and self.myPosSat is not None:
            I = np.clip(I, self.myNegSat, self.myPosSat)

        D = self.myKd * self.filtered_d
        PID = P + I + D

        if self.myNegSat is not None and self.myPosSat is not None:
            PID = np.clip(PID, self.myNegSat, self.myPosSat)

        self.myLastError = [error, t]
        return PID

    
    def response(self):
        y0 = [0.0, 0.0]  # Anfangswerte
        # Zeitbereich
        t_span = (0, 10)
        sol = solve_ivp(self._system_dynamics, t_span, y0, max_step=0.1, method='RK23', vectorized=False)
        timet = sol.t
        step_response = sol.y[self.order_i]

        itae_krit = self.__itae(timet,step_response, self.mySetPoint)
        return itae_krit, timet, step_response
    
    @staticmethod
    def __itae(t, y, setPoint):
        value = 0
        t_alt = 0
        for r in range(len(t)):
            delta_t = t[r] - t_alt
            value += t[r] * abs((setPoint - y[r]) * delta_t)  # ITAE
            t_alt = t[r]
        error = value
        return error
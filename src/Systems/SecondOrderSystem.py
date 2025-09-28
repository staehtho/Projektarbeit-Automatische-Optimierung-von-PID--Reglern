# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:33:20 2024

@author: lukas
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SecondOrderSystem:
    def __init__(self, Kp, Ki=None, Kd=None, Ti=None, Td=None, pos_sat=None, neg_sat=None, omega_0=1.0, D=0.3, K=1.0, setPoint=1.0):
        self.myKp = Kp
        if Ki is not None:
            self.myKi = Ki
        else:
            self.myKi = Kp / Ti
            
        if Kd is not None:
            self.myKd = Ki
        else:
            self.myKd = Kp * Td
            
        self.myPosSat = pos_sat
        self.myNegSat = neg_sat
        self.myIntegralError = 0.0
        self.myLastError = [0.0, 0.0]
        self.myOmega2d = -2*D*omega_0
        self.myOmega2 = -1*omega_0**2
        self.prev_d = 0
        self.mySetPoint = setPoint
        self.filtered_d = 0
        self.tau = 0.01
        self.order_i=0
        self.Ks = 1
        self.T = 1
        self.D = 0.3
    
    def __pid_controller(self, t, y):
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
    
    def __system_dynamics(self, t, y):
        Ks = self.Ks
        T = self.T
        D = self.D
        
        y1, y2 = y
        u = self.__pid_controller(t, y)
        dy1dt = y2
        dy2dt = (Ks * u - y1 - 2 * D * T * y2) / T**2
        return [dy1dt, dy2dt]
    
    def response(self):
        y0 = [0.0, 0.0]  # Anfangswerte
        # Zeitbereich
        t_span = (0, 20)
        sol = solve_ivp(self.__system_dynamics, t_span, y0, max_step=0.01, method='RK45', vectorized=False)
        timet = sol.t
        step_response = sol.y[self.order_i]

        itae_krit = self.__itae(timet,step_response)
        return timet, step_response, itae_krit
    
    @staticmethod
    def __itae(t, y):
        value = 0
        t_alt = 0
        for r in range(len(t)):
            delta_t = t[r] - t_alt
            value += t[r] * abs((1 - y[r]) * delta_t)  # ITAE
            t_alt = t[r]
        error = value
        return error
    
    @staticmethod
    def systemResponse(Kp, Ti, Td):
        system = SecondOrderSystem(Kp=Kp, Ti=Ti, Td=Td)
        itae_krit = system.response()
        return itae_krit
    
    @staticmethod
    def systemResponseX(X, pos_sat=5, neg_sat=-5):
        system = SecondOrderSystem(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat)
        timet, step_response, itae_krit = system.response()
        plt.plot(timet, step_response, color='blue', linewidth=2)
        textstr = f'$K_p = {X[0]}$\n$T_i = {X[1]}$\n$T_d = {X[2]}$'
        plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=18,verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.ylim(0,1.1)
        plt.legend()
        plt.grid(True)
        plt.show()
        return itae_krit

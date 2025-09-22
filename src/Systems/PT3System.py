# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:53:34 2024

@author: lukas
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class PT3System:
    def __init__(self, Kp, Ki=None, Kd=None, Ti=None, Td=None, pos_sat=None, neg_sat=None, omega_0=1.0, D=0.9, K=1.0, setPoint=1.0):
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
        self.order_i=2
        self.T1 = 1
        self.T2 = 1
        self.T3 = 1
        self.K = 1
        self.PIDTime = []
        self.PList = []
        self.IList = []
        self.DList = []
        self.PIDList = []
    
    def __pid_controller(self, t, y):
        error = self.mySetPoint - y[self.order_i]
        dt = t - self.myLastError[1]

        if dt > 0:
            error_diff = error - self.myLastError[0]
            derivative = error_diff / dt
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
        T1 = self.T1
        T2 = self.T2
        T3 = self.T3
        K = self.K
        
        y1, y2, y3 = y
        u = self.__pid_controller(t, y)
        dy1dt = (K * u - y1) / T1
        dy2dt = (y1 - y2) / T2
        dy3dt = (y2 - y3) / T3
        return [dy1dt, dy2dt, dy3dt]
    
    def response(self):
        y0 = [0.0, 0.0, 0.0]  # Anfangsauslenkung y(0) und Anfangsgeschwindigkeit y'(0)
        # Zeitbereich
        t_span = (0, 12)
        sol = solve_ivp(self.__system_dynamics, t_span, y0, max_step=0.01, method='RK45', vectorized=False)
        timet = sol.t
        step_response = sol.y[self.order_i]

        itae_krit = self.__itae(timet,step_response)
        return itae_krit, timet, step_response
    
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
        system = PT3System(Kp=Kp, Ti=Ti, Td=Td)
        itae_krit = system.response()
        return itae_krit
    
    @staticmethod
    def systemResponseX(X, pos_sat=10, neg_sat=-10):
        system = PT3System(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat)
        itae_krit = system.response()[0]
        return itae_krit
    
    @staticmethod
    def plotSystemResponseX(X, pos_sat=10, neg_sat=-10):
        system = PT3System(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat)
        itae_krit, timet, step_response = system.response()
        plt.plot(timet, step_response, color='blue', linewidth=2)
        textstr = f'$K_p = {X[0]}$\n$T_i = {X[1]}$\n$T_d = {X[2]}$'
        plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=18,verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.ylim(0,1.3)
        plt.legend()
        plt.grid(True)
        plt.show()
        print(itae_krit)
        
    @staticmethod
    def plotPID(X, pos_sat=2, neg_sat=-2):
        system = PT3System(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat)
        itae_krit, timet, step_response = system.response()
        plt.plot(timet, step_response, color='blue', linewidth=2, label='Sprungantwort')
        textstr = f'$K_p = {X[0]}$\n$T_i = {X[1]}$\n$T_d = {X[2]}$'
        plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=18,verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()
        plt.grid(True)
        plt.show()
        print(itae_krit)
        
# X represents the 3 controll parameter Kp, Ti, Td.
PT3System.plotSystemResponseX(X=[10,20,0.7])
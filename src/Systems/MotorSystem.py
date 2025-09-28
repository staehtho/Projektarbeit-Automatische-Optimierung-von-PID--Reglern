# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:22:37 2024

@author: lukas
"""

import matplotlib.pyplot as plt
from .closedLoopSystem import ClosedLoopSystem


class MotorSystem(ClosedLoopSystem):
    def __init__(self, Kp, Ki=None, Kd=None, Ti=None, Td=None, pos_sat=None, neg_sat=None, K=1.0, _KI=1.061, _TI=1.257, setPoint=100.0, tau_d=0.01, control=True):
        super().__init__(Kp, Ki, Kd, Ti, Td, pos_sat, neg_sat, K, setPoint, tau_d, control=control)
        # Subclass Constructor
        self.K_I = _KI
        self.T_I = _TI
    
    def _system_dynamics(self, t, y):
        x1, x2 = y
        u = self._pid_controller(t, y)
        dx1dt = x2
        dx2dt = -1/self.T_I * x2 + self.K_I/self.T_I * u
        return [dx1dt, dx2dt]
    
    @staticmethod
    def systemResponse(Kp, Ti, Td):
        system = MotorSystem(Kp=Kp, Ti=Ti, Td=Td)
        itae_krit = system.response()
        return itae_krit
    
    @staticmethod
    def systemResponseX(X, pos_sat=None, neg_sat=None):
        system = MotorSystem(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat)
        itae_krit = system.response()[0]
        return itae_krit
    
    @staticmethod
    def plotSystemResponseX(X, pos_sat=48, neg_sat=-48, control=True):
        system = MotorSystem(Kp=X[0], Ti=X[1], Td=X[2], pos_sat=pos_sat, neg_sat=neg_sat, control=control)
        itae_krit, timet, step_response = system.response()
        plt.plot(timet, step_response, color='blue', linewidth=2)
        textstr = f'$K_p = {X[0]}$\n$T_i = {X[1]}$\n$T_d = {X[2]}$'
        plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=18,verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
        plt.title('Werkzeugschlitten Motor')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Weg (mm)')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(itae_krit)
        
    @staticmethod
    def plotSystemResponseX2(X, pos_sat=48, neg_sat=-48, control=True):
        system = MotorSystem(Kp=X[0], Ki=X[1], Kd=X[2], pos_sat=pos_sat, neg_sat=neg_sat, control=control)
        itae_krit, timet, step_response = system.response()
        plt.plot(timet, step_response, label='step response')
        plt.title('Werkzeugschlitten Motor')
        plt.xlabel('Zeit (s)')
        plt.ylabel('Weg (mm)')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(itae_krit)

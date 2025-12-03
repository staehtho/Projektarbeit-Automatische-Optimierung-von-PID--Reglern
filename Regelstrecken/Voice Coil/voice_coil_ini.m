clc, clear variables
format compact
%% model

% parameters
km = 12.9;   % current-force-constant (N/A) (datasheet)
ru = 6.8e-3; % radius of unbalanced mass (m)
mu = 22e-3;  % unbalanced mass (kg)


%% time-domain system identification

% step response measurement
% - 0.5 A step starting at 0.1 sec
% - measurement time 1.0 sec
load data_00.mat % save step_meas_00 data_meas

figure(1)
subplot(211)
plot(data_meas.time, data_meas.signals(1).values), grid on
ylabel('Input Current (A)'), xlabel('Time (sec)')
subplot(212)
plot(data_meas.time, data_meas.signals(2).values), grid on
ylabel('Output Position (mm)'), xlabel('Time (sec)')

Ks = 10.2;
Tp = 0.117; % be aware of the time shift here
d1 = 3.8;
d2 = 3;
theta = log(d1 / d2);
D = 1 / sqrt(1 + pi^2/theta^2)
wd = 2*pi / Tp;
w0 = wd / sqrt(1 - D^2)
% 
% % identified model (time-domain)
s = tf('s');
Gvc_mod1 = Ks * w0^2 / (s^2 + 2*D*w0*s + w0^2);
% 
% % step response simulation (MATLAB)
y_mod = lsim(Gvc_mod1, ...
     data_meas.signals(1).values, ...
     data_meas.time);
% 
 figure(2)
 plot(data_meas.time, [data_meas.signals(2).values, y_mod]), grid on
 ylabel('Output Position (mm)'), xlabel('Time (sec)')
 legend('Measurement', 'Simulation', ...
     'Location', 'best')

 %% 
load('chirpmeas.mat')

Ks2 = 10;
D2 = 0.1;
w02 = 48;
% 
% % identified model (time-domain)
Gvc_mod2 = Ks2 * w02^2 / (s^2 + 2*D2*w02*s + w02^2);
grid on;
 bode(G, Gvc_mod1, Gvc_mod2)
 legend('Measurement', 'Simulation step', 'Simulation chirp',...
     'Location', 'best')

 %%
 Dcl = 1;
 w0 = 48;
 wcl = 4*w0;
 Ks = 10;
 D = 0.1;



Kp = -(w0^2 - wcl^2)/(Ks*w0^2);
Kd = -(2*(D*w0 - Dcl*wcl))/(Ks*w0^2);

Kw = Ks*Kp / (1+Ks*Kp);

Tv = Kd/Kp;
Tf = Tv/10;

Gr = Kp * (Tv*s + 1)/(Tf*s + 1);

%% P5 startet hier:

w0_P5 = w0;
wm_P5 = 2.3 * w0_P5;

D_P5 = D;
Ks_P5 = Ks;

Kd_P5 = ((7 * wm_P5) / w0_P5 - 2 * D_P5) * 1 / (Ks_P5 * w0_P5);
Kp_P5 = ((11 * wm_P5^2) / w0_P5^2 - 1) * 1 / (Ks_P5);
Ki_P5 = (5 * wm_P5^3) / (Ks_P5 * w0_P5^2);

Tn_P5 = Kp_P5 / Ki_P5;
Tv_P5 = Kd_P5 / Kp_P5;
Tf_P5 = Tv_P5 / 10;

%% Vergleich PID und PID-T1

G_PID = Kp_P5 * (1 + 1 / (Tn_P5 * s) + Tv_P5 * s);
G_PID_T1 = G_PID / (Tf_P5 * s + 1);

figure(10)
subplot(121)
bode(G_PID, G_PID_T1)
legend("PID", "PID-T1")
grid
subplot(122)
pzmap(G_PID, G_PID_T1)
legend("PID", "PID-T1")
grid

%%
Gv = 1 / (Tn_P5 * Tv_P5 * s^2 + Tn_P5 * s + 1);

%% Anregung von Unwucht

km_P5 = 12.9;
mu_P5 = 0.022;
ru_P5 = 6.8/1000;

load date_meas_01.mat % save date_meas_01 data_meas

y_mod = lsim(G_PID_T1, ...
     data_meas.signals(1).values, ...
     data_meas.time);

figure(20)
 plot(data_meas.time, [data_meas.signals(2).values, y_mod]), grid on
 ylabel('Output Position (mm)'), xlabel('Time (sec)')
 legend('Measurement', 'Simulation', ...
     'Location', 'best')


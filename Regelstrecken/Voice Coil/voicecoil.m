clc, clear variables
s = tf('s');

%% System

Ks = 10;
D = 0.1;
w0 = 48;
% 
G_voicecoil = Ks * w0^2 / (s^2 + 2*D*w0*s + w0^2);

%% Referenzwerte
Kp = 4.92;
Ti = 19.2e-3;
Td = 5.7e-3;
Tf = 0.57e-3;;
filter = s / (Tf * s + 1);
C_ref = Kp * (Td * filter + 1 + 1/(Ti*s))

%% PSO-Werte
Kp = 9.79;
Ti = 0.49;
Td = 0.005;
Tf = 1.21e-3 / 50;
filter = s / (Tf * s + 1);
C_pso = Kp * (Td * filter + 1 + 1/(Ti*s))

%% Sensitivitäten & Übertragungsfunktionen
L_ref = C_ref * G_voicecoil;
L_pso = C_pso * G_voicecoil;

S_ref = feedback(1, L_ref);
S_pso = feedback(1, L_pso);

T_ref = 1 - S_ref;
T_pso = 1 - S_pso;

SP_ref = G_voicecoil * S_ref;
SP_pso = G_voicecoil * S_pso;

% 80 – Regler
figure(80); clf;
bode(C_ref, C_pso), grid on
legend('Referenzparameter','PSO-Parameter')
title('Regler C')

% 81 – Loop Gain L
figure(81); clf;
bode(L_ref, L_pso), grid on
legend('Referenzparameter','PSO-Parameter')
title('Open Loop L')

% 82 – Sensitivity S
figure(82); clf;
bode(S_ref, S_pso), grid on
legend('Referenzparameter','PSO-Parameter')
title('Sensitivität S')

% 83 – Komplementäre Sensitivität T
figure(83); clf;
bode(T_ref, T_pso), grid on
legend('Referenzparameter','PSO-Parameter')
title('Führungsübertragungsfunktion T')


%% Plot
ref  = load('step_meas_ref.mat'); % save step_meas_ref data
pso  = load('step_meas_pso.mat'); % save step_meas_pso data

figure(1); clf;

% --- Einheitssprung (einmal aus REF) ---
plot(ref.data.time, ref.data.signals(1).values, 'k', 'LineWidth', 1.5);
hold on;

% --- Position PSO ---
plot(pso.data.time, pso.data.signals(2).values, 'b', 'LineWidth', 1.5);

% --- Position REF ---
plot(ref.data.time, ref.data.signals(2).values, 'r--', 'LineWidth', 1.5);

ylim([-0.2 1.4]);
grid on;

xlabel('Time (sec)');
ylabel('Position (mm)');
legend('Einheitsschritt', 'PSO-Parameter', 'Referenzparameter');

figure(2); clf;

% PSO-Stellgrösse
plot(pso.data.time, pso.data.signals(3).values, 'b', 'LineWidth', 1.5);
hold on;

% Einheitssprung (einmal aus REF)
plot(ref.data.time, ref.data.signals(1).values, 'k', 'LineWidth', 2);

ylim([-3 3]);
grid on;

xlabel('Time (sec)');
ylabel(' Strom (A)');
legend('Einheitsschritt', 'Stellgrösse Reglerausgang');

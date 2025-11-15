% Simulation eines schwingungsfähigen PT2-Systems mit variabler Dämpfung D
clear; clc;

% Numerator (immer gleich)
num = [1];

% 9 verschiedene Dämpfungswerte (unterkritisch -> kritisch -> überkritisch)
D_list = [1 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0];

% Limitierungen
limits = [
    2  -2;
    3  -3;
    5  -5;
    10 -10
];

% Lade PID-Parameter
run('PT2_parameter_PSO.m')

% Ergebnisliste vorbereiten
results = [];

% Hauptschleifen
for i = 1:length(D_list)
    D = D_list(i);

    % Nenner nach PT2-Formel (TF-Block liest num und den)
    den = [1, 2*D, 1];

    for j = 1:size(limits,1)
        upper_limit = limits(j,1);
        lower_limit = limits(j,2);

        % PID-Werte aus Tabelle holen
        Kp = pid_params{i,j}(1);
        Tn = pid_params{i,j}(2);
        Tv = pid_params{i,j}(3);

        % Zeitmessung starten
        t_start = tic;

        % Simulink-Modell starten (verwende Variablen aus aktuellem Workspace)
        sim('PT2_schwingfaehig_model','SrcWorkspace','current');

        % Simulationszeit stoppen
        sim_duration = toc(t_start);

        % Simulationsergebnis laden
        t = ans.simout.time;
        x = ans.simout.signals.values;

        % ITAE berechnen
        ITAE = 0;
        t_alt = 0;
        for r = 1:length(t)
            delta_t = t(r) - t_alt;
            ITAE = ITAE + t(r) * abs((1 - x(r))) * delta_t;
            t_alt = t(r);
        end

        % Ergebnis speichern
        results = [results; i, D, j, Kp, Tn, Tv, upper_limit, lower_limit, ITAE, sim_duration];

        % Fortschritt ausgeben
        fprintf('D = %.2f | Limit %d | Kp=%.2f, Tn=%.2f, Tv=%.2f → ITAE= %.6f | Dauer = %.2f s\n', ...
            D, j, Kp, Tn, Tv, ITAE, sim_duration);
    end
end

% Ergebnisse in Tabelle
T_out = array2table(results, ...
    'VariableNames', {'Index','Daempfung','LimitIndex','Kp','Tn','Tv','UpperLimit','LowerLimit','ITAE','SimTime_s'});

% CSV speichern
writetable(T_out, 'PT2_PSO_results_matlab.csv');

disp('✅ Alle Simulationen abgeschlossen. Ergebnisse in "PT2_PSO_results_matlab.csv" gespeichert.');

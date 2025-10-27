% Automatisierte Simulation mehrerer PT2-Systeme mit individuellen PID-Parametern pro Limitierung
clear; clc;

% Numerator (immer gleich)
num = [1];

% Verschiedene PT2-Systeme (Nenner)
den_list = {
    [1 2 1];
    [1 3 2];
    [1 1.5 0.5];
    [1 0.5 0.2];
    [1 4 3];
    [1 0.8 0.1];
};

% Limitierungen (4 Stück)
limits = [
    1  -1;
    2  -2;
    5  -5;
    10 -10
];

% PID-Parameter pro System und Limit
% Format: pid_params{i,j} = [Kp, Tn, Tv];
pid_params = cell(length(den_list), size(limits,1));

% PID-Parameter je System und Limit (Ks=1, T=1)
% Format: pid_params{i,j} = [Kp, Tn, Tv];

% PT1
pid_params{1,1} = [9.3, 2.9, 0];
pid_params{1,2} = [9.5, 1.9, 0];
pid_params{1,3} = [9.1, 1.2, 0];
pid_params{1,4} = [10, 1.0, 0];

% PT2
pid_params{2,1} = [10, 9.6, 0.3];
pid_params{2,2} = [10, 7.3, 0.3];
pid_params{2,3} = [9.6, 5.4, 0.3];
pid_params{2,4} = [9.8, 4.7, 0.3];

% PT3
pid_params{3,1} = [5.4, 9.4, 0.7];
pid_params{3,2} = [7.0, 10.0, 0.7];
pid_params{3,3} = [8.2, 9.6, 0.7];
pid_params{3,4} = [10.0, 9.7, 0.7];

% PT4
pid_params{4,1} = [1.9, 5.0, 1.1];
pid_params{4,2} = [2.4, 5.9, 1.2];
pid_params{4,3} = [2.3, 5.7, 1.2];
pid_params{4,4} = [2.1, 5.0, 1.1];

% PT5
pid_params{5,1} = [1.4, 5.3, 1.4];
pid_params{5,2} = [1.4, 5.2, 1.4];
pid_params{5,3} = [1.4, 5.2, 1.4];
pid_params{5,4} = [1.4, 5.0, 1.4];

% PT6
pid_params{6,1} = [1.1, 5.5, 1.7];
pid_params{6,2} = [1.1, 5.5, 1.7];
pid_params{6,3} = [1.1, 5.4, 1.7];
pid_params{6,4} = [1.1, 5.3, 1.7];


% Ergebnisliste vorbereiten
results = [];

% Hauptschleifen
for i = 1:length(den_list)
    den = den_list{i};

    for j = 1:size(limits,1)
        upper_limit = limits(j,1);
        lower_limit = limits(j,2);

        % PID-Werte aus Matrix holen
        Kp = pid_params{i,j}(1);
        Tn = pid_params{i,j}(2);
        Tv = pid_params{i,j}(3);

        % Simulink-Modell starten (verwende Variablen aus aktuellem Workspace)
        sim('PTn_closedloop_model','SrcWorkspace','current');

        % Signale extrahieren
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
        results = [results; i, j, Kp, Tn, Tv, upper_limit, lower_limit, ITAE];

        % Fortschritt anzeigen
        fprintf('PT%d | Limit %d | Kp=%.2f, Tn=%.2f, Tv=%.2f → ITAE =%.6f\n', ...
            i, j, Kp, Tn, Tv, ITAE);
    end
end

% Ergebnisse in Tabelle
T = array2table(results, ...
    'VariableNames', {'SystemIndex','LimitIndex','Kp','Tn','Tv','UpperLimit','LowerLimit','ITAE'});

% CSV speichern
writetable(T, 'PTn_simulation_results.csv');

disp('✅ Alle Simulationen abgeschlossen. Ergebnisse in "simulation_results.csv" gespeichert.');

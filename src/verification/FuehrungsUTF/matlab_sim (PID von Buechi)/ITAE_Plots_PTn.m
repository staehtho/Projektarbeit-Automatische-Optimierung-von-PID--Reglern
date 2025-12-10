% Automatisierte Simulation mehrerer PT2-Systeme mit individuellen PID-Parametern pro Limitierung
clear; clc;

% Numerator (immer gleich)
num = [1];

% Verschiedene PT2-Systeme (Nenner)
den_list = {
    [1 1];    
    [1 2 1];
    [1 3 3 1];
    [1 4 6 4 1];
    [1 5 10 10 5 1];
    [1 6 15 20 15 6 1];
};

% Limitierungen (4 Stück)
limits = [
    2  -2;
    3  -3;
    5  -5;
    10 -10
];

% Lade PID-Parameter
run('PTn_parameter_PSO.m')

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
        sim('PTn_model','SrcWorkspace','current');

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
writetable(T, 'PTn_PSO_results_matlab.csv');

disp('Alle Simulationen abgeschlossen. Ergebnisse in "simulation_results.csv" gespeichert.');

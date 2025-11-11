
% PID-Parameter je System und Limit (Ks=1, T=1)
% Format: pid_params{i,j} = [Kp, Tn, Tv];
pid_params = cell(6, 4);

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
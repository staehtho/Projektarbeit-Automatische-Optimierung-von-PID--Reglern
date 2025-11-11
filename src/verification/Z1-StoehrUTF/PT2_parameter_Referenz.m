% === PID-Parameter für PT2 schwingungsfähig ===
% Format: pid_params{i,j} = [Kp, Tn, Tv];
% Ks = 1, T = 1

% D = 1.0
pid_params{1,1} = [10, 9.6, 0.3];
pid_params{1,2} = [10, 7.3, 0.3];
pid_params{1,3} = [9.6, 5.4, 0.3];
pid_params{1,4} = [9.8, 4.7, 0.3];

% D = 0.7
pid_params{2,1} = [10, 8.6, 0.35];
pid_params{2,2} = [10, 6.8, 0.35];
pid_params{2,3} = [10, 5.4, 0.35];
pid_params{2,4} = [9.9, 4.6, 0.35];

% D = 0.6
pid_params{3,1} = [9.8, 8.3, 0.4];
pid_params{3,2} = [10, 6.9, 0.4];
pid_params{3,3} = [10, 5.2, 0.35];
pid_params{3,4} = [9.9, 4.9, 0.4];

% D = 0.5
pid_params{4,1} = [9.9, 8.1, 0.4];
pid_params{4,2} = [9.8, 6.5, 0.4];
pid_params{4,3} = [9.8, 5.3, 0.4];
pid_params{4,4} = [9.9, 4.7, 0.4];

% D = 0.4
pid_params{5,1} = [9.7, 7.6, 0.4];
pid_params{5,2} = [10, 6.4, 0.4];
pid_params{5,3} = [10, 5.2, 0.4];
pid_params{5,4} = [9.9, 4.5, 0.4];

% D = 0.3
pid_params{6,1} = [9.4, 7.3, 0.45];
pid_params{6,2} = [9.7, 6.3, 0.45];
pid_params{6,3} = [9.9, 5.4, 0.45];
pid_params{6,4} = [9.9, 4.8, 0.45];

% D = 0.2
pid_params{7,1} = [9.7, 7.3, 0.45];
pid_params{7,2} = [9.9, 6.2, 0.45];
pid_params{7,3} = [9.9, 5.2, 0.45];
pid_params{7,4} = [9.9, 4.6, 0.45];

% D = 0.1
pid_params{8,1} = [9.9, 7.5, 0.5];
pid_params{8,2} = [9.8, 6.3, 0.5];
pid_params{8,3} = [10, 5.5, 0.5];
pid_params{8,4} = [9.9, 4.9, 0.5];

% D = 0.0
pid_params{9,1} = [10, 7.3, 0.5];
pid_params{9,2} = [10, 6.2, 0.5];
pid_params{9,3} = [10, 5.3, 0.5];
pid_params{9,4} = [9.9, 4.7, 0.5];
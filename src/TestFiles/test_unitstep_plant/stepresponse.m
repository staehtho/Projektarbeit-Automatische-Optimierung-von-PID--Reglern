%%
%% 
load step_sim_00.mat % save('step_sim_00.mat', 'out');

%% transfer function
s = tf('s');
G = 1 / (s^2 + 0.6*s + 1);

%% ITEA
value = 0.0;
t_old = 0.0;

for i = 1:length(out.tout)
    ti = out.tout(i);
    yi = out.yout.signals(2).values(i);
    delta_t = ti - t_old;
    value = value + ti * abs((1 - yi) * delta_t);
    t_old = ti;
end

format long
disp(value)

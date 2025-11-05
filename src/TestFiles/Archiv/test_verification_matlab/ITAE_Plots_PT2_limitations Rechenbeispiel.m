clear

%PT2_2
format long

Kp = 10;
Tn = 9.6;
Tv = 0.3;

sim('PT2_Param_limited_Matlab') 

t_PT2_2 = simout.time;
x_PT2_2 = simout.signals.values;

plot(t_PT2_2,x_PT2_2);

ITAE_PT2_2 = 0;
t_alt = 0;
     for r = 1:length(t_PT2_2);
          delta_t = t_PT2_2(r)-t_alt;
          %IAE(za,zb) = IAE(za,zb) + abs((1-x(r))*delta_t);
          %ISE(za,zb) = ISE(za,zb) + (1-x(r))^2*delta_t;
          ITAE_PT2_2 = ITAE_PT2_2 + t_PT2_2(r)*abs((1-x_PT2_2(r))*delta_t); %isch da richtig?
          t_alt = t_PT2_2(r);
             
      end; 
            

ITAE_PT2_2
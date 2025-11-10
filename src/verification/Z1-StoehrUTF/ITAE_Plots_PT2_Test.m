clear

%PT2_2

num = [1];
den = [1 2 1];

Kp = 10;
Tn = 9.6;
Tv = 0.3;
upper_limit = 2;
lower_limit = -2;

sim('PTn_model.slx') 

t_PT2_2 = simout.time;
x_PT2_2 = simout.signals.values;

plot(t_PT2_2,x_PT2_2);

ITAE_PT2_2 = 0;
t_alt = 0;
     for r = 1:length(t_PT2_2);
          delta_t = t_PT2_2(r)-t_alt;
          ITAE_PT2_2 = ITAE_PT2_2 + t_PT2_2(r)*abs((1-x_PT2_2(r))*delta_t); %isch da richtig?
          t_alt = t_PT2_2(r);
             
      end; 
            
ITAE_PT2_2
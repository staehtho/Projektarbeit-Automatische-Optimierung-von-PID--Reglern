clear

s = tf('s');

%gegebene Parameter
%Kp = 4.92;
%Ti = 19.2e-3;
%Td = 05.7e-3;
%Tf = 0.57e-3;

%pso parameter
%Kp = 2.2952;
%Ti = 0.0993;
%Td = 0.006;
%Tf = 1/2000;

%pso parameter
Kp = 2.3;
Ti = 0.1;
Td = 0.006;
Tf = 1/2000;

Filter = s / (Tf * s + 1);

sim('voicecoil') 

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
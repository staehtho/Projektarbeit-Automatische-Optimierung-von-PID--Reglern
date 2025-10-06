clear

%PT2_2

Kp = 10;
Tn = 9.6;
Tv = 0.3
 
sim('PID_PT2_Param_limited_2') 
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

%PT2_3

Kp = 10
Tn = 7.3
Tv = 0.3
 
sim('PID_PT2_Param_limited_3') 
t_PT2_3 = simout.time;
x_PT2_3 = simout.signals.values;

plot(t_PT2_3,x_PT2_3);

ITAE_PT2_3 = 0;
t_alt = 0;
     for r = 1:length(t_PT2_3);
          delta_t = t_PT2_3(r)-t_alt;
          %IAE(za,zb) = IAE(za,zb) + abs((1-x(r))*delta_t);
          %ISE(za,zb) = ISE(za,zb) + (1-x(r))^2*delta_t;
          ITAE_PT2_3 = ITAE_PT2_3 + t_PT2_3(r)*abs((1-x_PT2_3(r))*delta_t); %isch da richtig?
          t_alt = t_PT2_3(r);
             
      end; 
            
ITAE_PT2_3

%PT2_5

Kp = 9.6
Tn = 5.4
Tv = 0.3
 
sim('PID_PT2_Param_limited_5') 
t_PT2_5 = simout.time;
x_PT2_5 = simout.signals.values;

plot(t_PT2_5,x_PT2_5);

ITAE_PT2_5 = 0;
t_alt = 0;
     for r = 1:length(t_PT2_5);
          delta_t = t_PT2_5(r)-t_alt;
          %IAE(za,zb) = IAE(za,zb) + abs((1-x(r))*delta_t);
          %ISE(za,zb) = ISE(za,zb) + (1-x(r))^2*delta_t;
          ITAE_PT2_5 = ITAE_PT2_5 + t_PT2_5(r)*abs((1-x_PT2_5(r))*delta_t); %isch da richtig?
          t_alt = t_PT2_5(r);
             
      end; 
            
ITAE_PT2_5

%PT2_10

Kp = 9.8
Tn = 4.7
Tv = 0.3
 
sim('PID_PT2_Param_limited_10') 
t_PT2_10 = simout.time;
x_PT2_10 = simout.signals.values;

plot(t_PT2_10,x_PT2_10);

ITAE_PT2_10 = 0;
t_alt = 0;
     for r = 1:length(t_PT2_10);
          delta_t = t_PT2_10(r)-t_alt;
          %IAE(za,zb) = IAE(za,zb) + abs((1-x(r))*delta_t);
          %ISE(za,zb) = ISE(za,zb) + (1-x(r))^2*delta_t;
          ITAE_PT2_10 = ITAE_PT2_10 + t_PT2_10(r)*abs((1-x_PT2_10(r))*delta_t); %isch da richtig?
          t_alt = t_PT2_10(r);
             
      end; 
            
ITAE_PT2_10


plot(t_PT2_2,x_PT2_2,'-k',t_PT2_3,x_PT2_3,'--k',t_PT2_5,x_PT2_5,'-.k',t_PT2_10,x_PT2_10,'.k','LineWidth',2);grid
legend({'controller output limitation +/-2','controller output limitation +/-3','controller output limitation +/-5','controller output limitation +/-10'},'Location','southeast')

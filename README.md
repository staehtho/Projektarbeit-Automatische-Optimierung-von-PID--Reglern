# Projektarbeit-Automatische-Optimierung-von-PID--Reglern

In this project, a fully Python-based open-source tool for the automatic tuning
of PID controllers for linear time-invariant systems is developed. The objective
is to systematically optimize the controller parameters $K_p$, $T_i$, and
$T_d$, thereby replacing classical tuning methods with a reproducible,
simulation-based optimization approach. The optimization is performed using
Particle Swarm Optimization (PSO) in combination with time-domain performance
criteria such as IAE, ISE, ITAE, and ITSE.

The developed tool follows a modular and object-oriented design. It supports
arbitrary linear transfer functions, actuator saturation, and anti-windup
mechanisms, and allows for flexible configuration of different excitation
scenarios. The closed-loop simulation is carried out entirely in the time
domain. To improve computational performance, performance-critical components
are vectorized and accelerated using just-in-time compilation.

The results demonstrate that the developed *PID-Optimizer* yields stable
and high-quality controller parameters for various classes of systems. Compared
to classical tuning methods, lower performance index values and shorter settling
times are achieved. Through vectorization of the cost function and the use of
Numba, the computational time of the optimization process is significantly
reduced, enabling efficient execution of extensive PSO runs. The simulation
results and optimized parameters were systematically compared with
MATLAB/Simulink and show a high level of agreement.

Overall, the developed *PID-Optimizer* represents a powerful, reliable,
and openly accessible tool for automatic PID tuning, providing a
practice-oriented alternative to commercial control engineering software.
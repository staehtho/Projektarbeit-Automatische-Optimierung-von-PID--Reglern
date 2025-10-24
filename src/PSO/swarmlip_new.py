import random
import numpy as np
import copy
import math
from typing import Callable, List, Optional


# TODO: PSO ist auf 3 Dimensionen beschränkt -> anpassen für dynamische Dimensionen

class Particle:
    """Represents a single particle in the swarm.

    Attributes:
        position (List[float]): Current position of the particle.
        velocity (np.ndarray): Current velocity of the particle.
        cost (Optional[float]): Cost of the current position.
        pBest_cost (Optional[float]): Best cost the particle has achieved.
        pBest_position (Optional[List[float]]): Position corresponding to pBest_cost.
    """

    def __init__(self,
                 position: List[float],
                 velocity: np.ndarray,
                 cost: Optional[float] = None,
                 pBest_cost: Optional[float] = None,
                 pBest_position: Optional[List[float]] = None) -> None:
        """
        Initializes a Particle.

        Args:
            position (List[float]): Initial position of the particle.
            velocity (np.ndarray): Initial velocity of the particle.
            cost (Optional[float]): Initial cost (default None).
            pBest_cost (Optional[float]): Initial personal best cost (default None).
            pBest_position (Optional[List[float]]): Initial personal best position (default None).
        """
        self.position: List[float] = position
        self.velocity: np.ndarray = velocity
        self.cost: Optional[float] = cost
        self.pBest_cost: Optional[float] = pBest_cost
        self.pBest_position: Optional[List[float]] = pBest_position


class SwarmNew:
    """Represents a swarm of particles for Particle Swarm Optimization (PSO).

    Attributes:
        obj_func (Callable[[List[float]], float]): Objective function to minimize.
        particles (List[Particle]): List of particles in the swarm.
        gBest (Optional[Particle]): Particle with the global best cost.
        size (int): Number of particles.
        bounds (List[List[float]]): Bounds of the search space [[min,...], [max,...]].
        iterations (int): Number of iterations performed.
    """

    def __init__(self,
                 obj_func: Callable[[np.ndarray], np.ndarray],
                 size: int,
                 bounds: List[List[float]],
                 randomness: Optional[float] = 1.0,
                 u1: Optional[float] = 1.49,
                 u2: Optional[float] = 1.49,
                 InertiaRange: Optional[tuple[float]] = (0.1, 1.1),
                 InitialSwarmSpan: Optional[int] = 2000,
                 MinNeighborsFraction: Optional[float] = 0.25,
                 maxStall: Optional[int] = 15,
                 space_factor: Optional[float] = 0.001,
                 convergence_factor: Optional[float] = 1e-2) -> None:
        """
        Initializes a Swarm.

        Args:
            obj_func (Callable[[List[float]], float]): Objective function to minimize.
            size (int): Number of particles.
            bounds (List[List[float]]): Search space bounds [[min1,...], [max1,...]].
            randomness (Optional[float]): Randomness factor for velocity update.
            u1 (Optional[float]): Cognitive coefficient.
            u2 (Optional[float]): Social coefficient.
            InertiaRange (Optional[tuple[float]]): [min_inertia, max_inertia].
            InitialSwarmSpan (Optional[int]): Used to initialize particle positions.
            MinNeighborsFraction (Optional[float]): Fraction of swarm considered as neighbors.
            maxStall (Optional[int]): Number of iterations to consider convergence.
            space_factor (Optional[float]): Factor for minimal particle space convergence criterion.
            convergence_factor (Optional[float]): Factor for cost convergence criterion.
        """
        self.obj_func: Callable[[np.ndarray], np.ndarray] = obj_func
        self.particles: List[Particle] = []
        self.c: int = 0
        self.iterations: int = 0
        self.spaceFactor: float = space_factor
        self.convergenceFactor: float = convergence_factor
        self.size: int = size
        self.bounds: List[List[float]] = bounds
        self.randomness: float = randomness
        self.options: List[float] = [InertiaRange[1], u1, u2]
        self.inertiaRange: tuple[float] = InertiaRange
        self.gBest: Optional[Particle] = None
        self.maxStall: int = maxStall
        self.initialSwarmSpan: int = InitialSwarmSpan
        self.minNeighborsFraction: float = MinNeighborsFraction
        self.__init_swarm()

    def __init_swarm(self) -> None:
        """Initializes particle positions and velocities randomly."""
        self.particles.clear()
        self.minNeighborhoodSize: int = max(2, math.floor(self.size * self.minNeighborsFraction))
        self.N: int = random.randint(self.minNeighborhoodSize, self.size - 1)
        r: np.ndarray = np.subtract(self.bounds[1], self.bounds[0])
        self.speed_bounds: List[List[float]] = [[-r[0], -r[1], -r[2]], [r[0], r[1], r[2]]]
        span: List[float] = [self.speed_bounds[1][0] / self.initialSwarmSpan,
                             self.speed_bounds[1][1] / self.initialSwarmSpan,
                             self.speed_bounds[1][2] / self.initialSwarmSpan]
        for i in range(self.size):
            Kp = self.bounds[0][0] + span[0] * random.randint(0, self.initialSwarmSpan)
            Ti = self.bounds[0][1] + span[1] * random.randint(0, self.initialSwarmSpan)
            Td = self.bounds[0][2] + span[2] * random.randint(0, self.initialSwarmSpan)
            Kp_v = random.uniform(self.speed_bounds[0][0], self.speed_bounds[1][0])
            Ti_v = random.uniform(self.speed_bounds[0][1], self.speed_bounds[1][1])
            Td_v = random.uniform(self.speed_bounds[0][2], self.speed_bounds[1][2])
            position: List[float] = [Kp, Ti, Td]
            velocity: np.ndarray = np.array([Kp_v, Ti_v, Td_v])
            p: Particle = Particle(position, velocity)
            self.particles.append(p)

        self.__initCosts()
        self.__initGlobalBest()

    def __getCosts(self) -> np.ndarray:
        """Evaluates the objective function for all particles.

        Returns:
            List[float]: Costs corresponding to each particle's position.
        """
        positions: List[List[float]] = [p.position for p in self.particles]

        return self.obj_func(np.array(positions, dtype=np.float64))

    def __initCosts(self) -> None:
        """Initializes particle costs and personal bests."""
        positions: List[List[float]] = [p.position for p in self.particles]

        costs = self.obj_func(np.array(positions, dtype=np.float64))

        for particle, cost in zip(self.particles, costs):
            particle.cost = cost
            particle.pBest_cost = cost
            particle.pBest_position = copy.deepcopy(particle.position)

    def __initGlobalBest(self) -> None:
        """Initializes the global best particle."""
        best_p_init: Particle = min(self.particles, key=lambda p: p.pBest_cost)
        self.gBest = copy.deepcopy(best_p_init)

    def __updateVelocity(self, i: int, p: Particle) -> None:
        """Updates the velocity of a particle.

        Args:
            i (int): Index of the particle.
            p (Particle): The particle to update.
        """
        u1: float = random.uniform(1.0 - self.randomness, 1.0)
        u2: float = random.uniform(1.0 - self.randomness, 1.0)

        particle_indizes: List[int] = list(range(self.size))
        particle_indizes.remove(i)
        S: List[int] = random.sample(particle_indizes, self.N)
        neighbours: List[Particle] = [self.particles[j] for j in S]
        best_neighbour: Particle = min(neighbours, key=lambda pp: pp.pBest_cost)

        vec_gBest: np.ndarray = np.subtract(best_neighbour.pBest_position, p.position)
        vec_pBest: np.ndarray = np.subtract(p.pBest_position, p.position)
        p.velocity = np.add(np.add(p.velocity * self.options[0], vec_gBest * self.options[1] * u1),
                            vec_pBest * self.options[2] * u2)
        p.velocity = np.clip(p.velocity, self.speed_bounds[0], self.speed_bounds[1])

    def __updatePosition(self, p: Particle) -> None:
        """Updates the position of a particle within bounds.

        Args:
            p (Particle): The particle to update.
        """
        p.position = np.add(p.position, p.velocity)
        for j in range(len(p.position)):
            if p.position[j] >= self.bounds[1][j]:
                p.position[j] = self.bounds[1][j]
                p.velocity[j] = 0
            if p.position[j] < self.bounds[0][j]:
                p.position[j] = self.bounds[0][j]
                p.velocity[j] = 0

    def __iterate(self) -> None:
        """Performs one iteration of velocity and position updates and updates best values."""
        new_best: bool = False
        for i, p in enumerate(self.particles):
            self.__updateVelocity(i, p)
            self.__updatePosition(p)

        costs = self.__getCosts()

        for particle, cost in zip(self.particles, costs):
            particle.cost = cost
            if cost <= particle.pBest_cost:
                particle.pBest_cost = cost
                particle.pBest_position = copy.deepcopy(particle.position)
            if cost <= self.gBest.pBest_cost:
                self.gBest = copy.deepcopy(particle)
                new_best = True

        if new_best:
            self.c = max(0, self.c - 1)
            self.N = self.minNeighborhoodSize
            if self.c < 2:
                self.options[0] = 2 * self.options[0]
            elif self.c > 5:
                self.options[0] = self.options[0] / 2
            self.options[0] = np.clip(self.options[0], self.inertiaRange[0], self.inertiaRange[1])
        else:
            self.c += 1
            self.N = min(self.N + self.minNeighborhoodSize, self.size - 1)

    def __getParticleSpace(self) -> float:
        """Calculates the hypervolume occupied by particles in parameter space.

        Returns:
            float: Volume of the particle positions.
        """
        min_Kp: float = min(particle.position[0] for particle in self.particles)
        max_Kp: float = max(particle.position[0] for particle in self.particles)
        min_Ti: float = min(particle.position[1] for particle in self.particles)
        max_Ti: float = max(particle.position[1] for particle in self.particles)
        min_Td: float = min(particle.position[2] for particle in self.particles)
        max_Td: float = max(particle.position[2] for particle in self.particles)

        particle_space: float = (max_Kp - min_Kp) * (max_Ti - min_Ti) * (max_Td - min_Td)
        return particle_space

    def simulate_swarm(self, iterate_func: Optional[Callable[['Swarm', float], None]] = None) -> 'Swarm':
        """Runs the swarm optimization until convergence.

        Args:
            iterate_func (Optional[Callable[[Swarm, float], None]]): Optional callback called each iteration
                with the swarm and particle space percentage.

        Returns:
            Swarm: Self with updated global best particle.
        """
        swarm_state: List[float] = [self.gBest.pBest_cost]
        termination_criteria: bool = False
        space_criteria: bool = False
        initial_space: float = self.speed_bounds[1][0] * self.speed_bounds[1][1] * self.speed_bounds[1][2]
        while True:
            self.__iterate()
            self.iterations += 1
            swarm_state.append(self.gBest.pBest_cost)
            particle_space: float = self.__getParticleSpace()
            particle_space_rel: int = round((particle_space / initial_space) * 100)
            if iterate_func is not None:
                iterate_func(self, particle_space_rel)
            if particle_space < initial_space * self.spaceFactor:
                space_criteria = True
            if space_criteria:
                termination_criteria = True
            if len(swarm_state) > self.maxStall and 1 - (
                    swarm_state[-1] / swarm_state[-self.maxStall]) <= self.convergenceFactor:
                termination_criteria = True
            if termination_criteria:
                break
        return self

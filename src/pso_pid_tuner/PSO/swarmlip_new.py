import random
import numpy as np
import copy
import math
from typing import Callable, List, Optional


# ============================================================================
# Particle class
# ============================================================================
class Particle:
    """Represents a single particle in the swarm.

    Each particle keeps track of its position, velocity, current cost, and personal best.
    Class-level attributes store global parameters shared among all particles.
    """

    # Class-level (shared) attributes
    bounds: np.ndarray = None
    coefficients: List[float] = None  # [inertia, cognitive_coeff, social_coeff]
    randomness: float = 1.0
    speed_bounds: np.ndarray = None

    # -------------------------------------------------------------------------
    # Instance-level attributes
    # -------------------------------------------------------------------------
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Initializes a particle with a position and velocity.

        Args:
            position (np.ndarray): Initial position vector of the particle.
            velocity (np.ndarray): Initial velocity vector of the particle.
        """
        self.position = position
        self.velocity = velocity
        self.cost: Optional[float] = None
        self.p_best_cost: Optional[float] = None
        self.p_best_position: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    # Class configuration
    # -------------------------------------------------------------------------
    @classmethod
    def configure_class(cls,
                        bounds: np.ndarray,
                        coefficients: List[float],
                        randomness: float) -> None:
        """Configures shared parameters for all particles.

        Args:
            bounds (np.ndarray): Global bounds for all particles.
            coefficients (List[float]): List of coefficients [inertia, c1, c2].
            randomness (float): Random factor for velocity update.
        """
        cls.bounds = bounds
        cls.coefficients = coefficients
        cls.randomness = randomness
        r = np.subtract(bounds[1], bounds[0])
        cls.speed_bounds = np.array([-r, r]).T  # velocity limits per dimension

    # -------------------------------------------------------------------------
    # Particle dynamics
    # -------------------------------------------------------------------------
    def update_velocity(self,
                        swarm_particles: List['Particle'],
                        i: int,
                        N: int) -> None:
        """Updates the particle's velocity based on personal best and neighborhood best.

        Args:
            swarm_particles (List[Particle]): List of all particles in the swarm.
            i (int): Index of this particle in the swarm.
            N (int): Number of neighbors considered for local best.
        """
        # Random factors for stochastic velocity update
        u1 = random.uniform(1.0 - Particle.randomness, 1.0)
        u2 = random.uniform(1.0 - Particle.randomness, 1.0)

        # Select neighborhood indices (exclude self)
        indices = list(range(len(swarm_particles)))
        indices.remove(i)
        neighbors = random.sample(indices, N)
        best_neighbor = min([swarm_particles[j] for j in neighbors],
                            key=lambda pp: pp.p_best_cost)

        # Compute velocity contributions
        vec_g_best = best_neighbor.p_best_position - self.position
        vec_p_best = self.p_best_position - self.position

        inertia, c1, c2 = Particle.coefficients
        self.velocity = (
                self.velocity * inertia
                + vec_g_best * c1 * u1
                + vec_p_best * c2 * u2
        )

        # Clip velocity to bounds
        for j in range(len(self.position)):
            min_v, max_v = Particle.speed_bounds[j]
            self.velocity[j] = np.clip(self.velocity[j], min_v, max_v)

    def update_position(self) -> None:
        """Updates the particle's position based on velocity, enforcing bounds."""
        self.position += self.velocity
        for j in range(len(self.position)):
            if self.position[j] > Particle.bounds[1][j]:
                self.position[j] = Particle.bounds[1][j]
                self.velocity[j] = 0
            elif self.position[j] < Particle.bounds[0][j]:
                self.position[j] = Particle.bounds[0][j]
                self.velocity[j] = 0

    def update_best(self, cost: float) -> None:
        """Updates the personal best position if the current cost is better.

        Args:
            cost (float): Current cost value of the particle.
        """
        self.cost = cost
        if self.p_best_cost is None or cost < self.p_best_cost:
            self.p_best_cost = cost
            self.p_best_position = copy.deepcopy(self.position)


# ============================================================================
# Swarm class
# ============================================================================
class SwarmNew:
    """Particle Swarm Optimization (PSO) swarm manager.

    This class manages the entire swarm, including initialization, iteration,
    global best tracking, and convergence criteria.
    """

    def __init__(self,
                 obj_func: Callable[[np.ndarray], np.ndarray],
                 size: int,
                 param_number: int,
                 bounds: List[List[float]],
                 randomness: float = 1.0,
                 u1: float = 1.49,
                 u2: float = 1.49,
                 inertia_range: tuple[float, float] = (0.1, 1.1),
                 initial_swarm_span: int = 2000,
                 min_neighbors_fraction: float = 0.25,
                 max_stall: int = 15,
                 space_factor: float = 0.001,
                 convergence_factor: float = 1e-2) -> None:
        """Initializes the PSO swarm with given parameters.

        Args:
            obj_func (Callable[[np.ndarray], np.ndarray]): Objective function.
            size (int): Number of particles in the swarm.
            param_number (int): Number of parameters (dimensions) to optimize.
            bounds (List[List[float]]): Bounds for each parameter [[min, max], ...].
            randomness (float, optional): Random factor for velocity updates. Defaults to 1.0.
            u1 (float, optional): Cognitive coefficient. Defaults to 1.49.
            u2 (float, optional): Social coefficient. Defaults to 1.49.
            inertia_range (tuple[float, float], optional): Inertia weight range. Defaults to (0.1, 1.1).
            initial_swarm_span (int, optional): Initial span divisions for particle positions. Defaults to 2000.
            min_neighbors_fraction (float, optional): Minimum fraction of swarm considered neighbors. Defaults to 0.25.
            max_stall (int, optional): Maximum iterations with little improvement. Defaults to 15.
            space_factor (float, optional): Factor to define particle space convergence. Defaults to 0.001.
            convergence_factor (float, optional): Relative change threshold for convergence. Defaults to 1e-2.
        """
        self.obj_func = obj_func
        self.size = size
        self.param_number = param_number
        self.bounds = np.array(bounds, dtype=float)
        self.randomness = randomness
        self._coefficients = [inertia_range[1], u1, u2]
        self._inertia_range = inertia_range
        self._max_stall = max_stall
        self._initial_swarm_span = initial_swarm_span
        self._min_neighbors_fraction = min_neighbors_fraction
        self._space_factor = space_factor
        self._convergence_factor = convergence_factor

        self.particles: List[Particle] = []
        self.gBest: Optional[Particle] = None
        self.iterations: int = 0
        self._no_improvement_counter = 0

        # Initialize swarm particles
        self._init_swarm()

    # -------------------------------------------------------------------------
    # Internal initialization
    # -------------------------------------------------------------------------
    def _init_swarm(self) -> None:
        """Initializes particle positions, velocities, and class-level parameters."""
        self.particles.clear()
        self._min_neighborhood_size = max(2, math.floor(self.size * self._min_neighbors_fraction))
        self._N = random.randint(self._min_neighborhood_size, self.size - 1)

        # Compute range and span for initialization
        r = np.subtract(self.bounds[1], self.bounds[0])
        span = r / self._initial_swarm_span

        # Configure shared particle parameters
        Particle.configure_class(
            bounds=self.bounds,
            coefficients=self._coefficients,
            randomness=self.randomness
        )

        # Initialize particles
        for _ in range(self.size):
            position = np.array([
                self.bounds[0][j] + span[j] * random.randint(0, self._initial_swarm_span)
                for j in range(self.param_number)
            ])
            velocity = np.array([random.uniform(-r[j], r[j]) for j in range(self.param_number)])
            self.particles.append(Particle(position, velocity))

        self._init_costs()
        self._init_global_best()

    def _get_costs(self) -> np.ndarray:
        """Evaluates the objective function for all particles.

        Returns:
            np.ndarray: Array of cost values for each particle.
        """
        positions = np.array([p.position for p in self.particles])
        return self.obj_func(positions)

    def _init_costs(self) -> None:
        """Initializes particle costs and personal best positions."""
        costs = self._get_costs()
        for particle, cost in zip(self.particles, costs):
            particle.update_best(cost)

    def _init_global_best(self) -> None:
        """Determines the initial global best particle in the swarm."""
        self.gBest = copy.deepcopy(min(self.particles, key=lambda p: p.p_best_cost))

        # -------------------------------------------------------------------------
        # Iteration
        # -------------------------------------------------------------------------

    def _iterate(self) -> None:
        """Performs a single iteration of the PSO algorithm.

        Updates velocities, positions, personal bests, and global best.
        Dynamically adjusts inertia and neighborhood size based on improvement.
        """
        new_best = False

        # Update velocity and position for all particles
        for i, p in enumerate(self.particles):
            p.update_velocity(self.particles, i, self._N)
            p.update_position()

        # Evaluate costs and update personal/global bests
        costs = self._get_costs()
        for particle, cost in zip(self.particles, costs):
            particle.update_best(cost)
            if particle.p_best_cost < self.gBest.p_best_cost:
                self.gBest = copy.deepcopy(particle)
                new_best = True

        # Adaptive neighborhood and inertia adjustments
        if new_best:
            self._no_improvement_counter = max(0, self._no_improvement_counter - 1)
            self._N = self._min_neighborhood_size
            if self._no_improvement_counter < 2:
                self._coefficients[0] *= 2
            elif self._no_improvement_counter > 5:
                self._coefficients[0] /= 2
            self._coefficients[0] = np.clip(
                self._coefficients[0],
                self._inertia_range[0],
                self._inertia_range[1]
            )
        else:
            self._no_improvement_counter += 1
            self._N = min(self._N + self._min_neighborhood_size, self.size - 1)

        # -------------------------------------------------------------------------
        # Utility
        # -------------------------------------------------------------------------

    def _get_particle_space(self) -> float:
        """Computes the hypervolume spanned by all particles in parameter space.

        Returns:
            float: Hypervolume spanned by particle positions.
        """
        min_positions = np.min([p.position for p in self.particles], axis=0)
        max_positions = np.max([p.position for p in self.particles], axis=0)
        return np.prod(max_positions - min_positions)

        # -------------------------------------------------------------------------
        # Public API
        # -------------------------------------------------------------------------

    def simulate_swarm(self,
                       iterate_func: Optional[Callable[['SwarmNew', float], None]] = None
                       ) -> 'SwarmNew':
        """Runs the PSO optimization until convergence criteria are met.

        Args:
            iterate_func (Optional[Callable[['SwarmNew', float], None]]):
                Optional callback executed each iteration. Receives the swarm
                and the particle space percentage.

        Returns:
            SwarmNew: The swarm instance after convergence.
        """
        swarm_state = [self.gBest.p_best_cost]
        termination_criteria = False
        space_criteria = False

        # Compute initial hypervolume for convergence comparison
        r = np.subtract(self.bounds[1], self.bounds[0])
        initial_space = np.prod(r)

        while True:
            self._iterate()
            self.iterations += 1
            swarm_state.append(self.gBest.p_best_cost)

            particle_space = self._get_particle_space()
            particle_space_rel = round((particle_space / initial_space) * 100)

            # Execute user-provided callback if available
            if iterate_func:
                iterate_func(self, particle_space_rel)

            # Check convergence based on particle hypervolume
            if particle_space < initial_space * self._space_factor:
                space_criteria = True
            if space_criteria:
                termination_criteria = True

            # Check convergence based on lack of improvement
            if len(swarm_state) > self._max_stall and 1 - (
                    swarm_state[-1] / swarm_state[-self._max_stall]
            ) <= self._convergence_factor:
                termination_criteria = True

            if termination_criteria:
                break

        return self

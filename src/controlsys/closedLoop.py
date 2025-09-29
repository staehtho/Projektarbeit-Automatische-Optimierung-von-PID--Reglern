from .plant import Plant


class ClosedLoop:
    """
    Represents a closed-loop control system with a PID controller.

    The closed-loop system connects a plant with a PID controller defined by
    proportional, integral, and derivative gains.

    Parameters
    ----------
    plant : Plant
        The plant or process to be controlled, an instance of the `Plant` class.
    kp : float
        Proportional gain of the PID controller.
    ki : float
        Integral gain of the PID controller.
    kd : float
        Derivative gain of the PID controller.

    Example
    -------
    Create a PID controlled closed-loop system:

        plant = Plant(num=[1], dec=[1, 2, 1])
        loop = ClosedLoop(plant=plant, kp=1.0, ki=0.5, kd=0.1)
        print(loop.kp)  # 1.0
    """

    def __init__(self,
                 plant: Plant,
                 kp: float,
                 ki: float,
                 kd: float
                 ) -> None:

        self._plant: Plant = plant
        self._kp = kp
        self._ki = ki
        self._kd = kd

    # ******************************
    # Attributes
    # ******************************

    @property
    def plant(self) -> Plant:
        """The plant (system) being controlled."""
        return self._plant

    @property
    def kp(self) -> float:
        """Proportional gain of the PID controller."""
        return self._kp

    @property
    def ki(self) -> float:
        """Integral gain of the PID controller."""
        return self._ki

    @property
    def kd(self) -> float:
        """Derivative gain of the PID controller."""
        return self._kd

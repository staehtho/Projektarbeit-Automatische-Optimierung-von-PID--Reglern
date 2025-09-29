import numpy as np

def laplace_2_response(num: list[float] | np.ndarray,
                       dec: list[float] |np.ndarray,
                       *,
                       solver_methode: str = "RK23",
                       max_step: float = 0.1
                       ) -> np.ndarray:

    inverse_laplace(num=np.array(num), dec=np.array(dec))

    response: np.ndarray = calculate_response(solver_methode=solver_methode, max_step=max_step)

    return response

def inverse_laplace(num: np.ndarray, dec: np.ndarray):
    pass

def calculate_response(solver_methode: str, max_step: float) -> np.ndarray:
    pass

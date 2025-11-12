import yaml
from pathlib import Path
import sys


class ConfigError(Exception):
    """Custom exception for invalid configuration."""
    pass


def load_config():
    if getattr(sys, "frozen", False):
        # exe läuft
        base_path = Path(sys.executable).parent.parent  # Ordner der exe
    else:
        base_path = Path(__file__).parent.parent

    # Config liegt in Unterordner 'config' neben exe oder im Projekt
    config_path = base_path / "config" / "config.yaml"
    errors = []  # Liste zur Sammlung aller Exceptions

    # YAML laden
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to load YAML: {e}")

    # --- SYSTEM ---
    system = cfg.get("system")
    if system is None or not isinstance(system, dict):
        errors.append("'system' section is missing or not a dictionary.")
        system = {}

    # Plant
    plant = system.get("plant")
    if plant is None or not isinstance(plant, dict):
        errors.append("'plant' section is missing or not a dictionary.")
        plant = {}

    numerator = plant.get("numerator")
    if not isinstance(numerator, list):
        errors.append(f"'numerator' should be a list, got {type(numerator).__name__}.")
    elif len(numerator) == 0:
        errors.append("'numerator' list must not be empty.")

    denominator = plant.get("denominator")
    if not isinstance(denominator, list):
        errors.append(f"'denominator' should be a list, got {type(denominator).__name__}.")
    elif len(denominator) == 0:
        errors.append("'denominator' list must not be empty.")

    # Simulation time
    sim_time = system.get("simulation_time", {})
    if not isinstance(sim_time, dict):
        errors.append("'simulation_time' section is missing or not a dictionary.")
        sim_time = {}

    for key in ["start_time", "end_time", "time_step"]:
        val = sim_time.get(key)

        # Versuche, Strings wie "1e-4" in float umzuwandeln
        if isinstance(val, str):
            try:
                val = float(val)
                sim_time[key] = val  # aktualisiere den Wert
            except ValueError:
                errors.append(f"'{key}' should be a number, got string '{val}' that cannot be converted.")
                continue

        if not isinstance(val, (int, float)):
            errors.append(f"'{key}' should be a number, got {type(val).__name__}.")

    # Anti-windup
    anti_windup = system.get("anti_windup")
    if anti_windup not in ["clamping", "conditional"]:
        errors.append(f"'anti_windup' should be 'clamping' or 'conditional', got {anti_windup}.")

    # excitation_target
    excitation_target = system.get("excitation_target")
    if excitation_target not in ["reference", "input_disturbance", "measurement_disturbance"]:
        errors.append(f"'excitation_target' should be 'reference' or 'input_disturbance' or "
                      f"'measurement_disturbance', got {excitation_target}.")

    # Control constraint
    constraint = system.get("control_constraint", {})
    min_val = constraint.get("min_constraint")
    max_val = constraint.get("max_constraint")

    # Typprüfung
    for key, val in [("min_constraint", min_val), ("max_constraint", max_val)]:
        if not isinstance(val, (int, float)):
            errors.append(f"'{key}' should be a number, got {type(val).__name__}.")

    # Logische Prüfung
    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
        if min_val >= max_val:
            errors.append(f"'min_constraint' ({min_val}) must be less than 'max_constraint' ({max_val}).")

    # --- PSO ---
    pso = cfg.get("pso", {})
    if not isinstance(pso, dict):
        errors.append("'pso' section is missing or not a dictionary.")
        pso = {}

    # PSO swarm size
    swarm_size = pso.get("swarm_size")
    if not isinstance(swarm_size, int):
        errors.append(f"'swarm_size' should be an integer, got {type(swarm_size).__name__}.")
    elif swarm_size <= 0:
        errors.append(f"'swarm_size' must be greater than 0, got {swarm_size}.")

    # PSO max
    iterations = pso.get("iterations")
    if not isinstance(iterations, int):
        errors.append(f"'max_iterations' should be an int, got {type(iterations).__name__}.")
    elif iterations <= 0:
        errors.append(f"'iterations' must be greater than 0, got {iterations}.")

    # PSO bounds
    bounds = pso.get("bounds", {})
    if not isinstance(bounds, dict):
        errors.append("'bounds' section is missing or not a dictionary.")
        bounds = {}

    # Typprüfung
    for key in ["kp_min", "kp_max", "ti_min", "ti_max", "td_min", "td_max"]:
        val = bounds.get(key)
        if not isinstance(val, (int, float)):
            errors.append(f"'{key}' should be a number, got {type(val).__name__}.")

    # Logische Prüfungen für min/max
    for param in ["kp", "ti", "td"]:
        min_val = bounds.get(f"{param}_min")
        max_val = bounds.get(f"{param}_max")

        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            if min_val >= max_val:
                errors.append(f"'{param}_min' ({min_val}) must be less than '{param}_max' ({max_val}).")

    # Spezielle Prüfung für ti_min > 0
    ti_min = bounds.get("ti_min")
    if isinstance(ti_min, (int, float)) and ti_min <= 0:
        errors.append(f"'ti_min' ({ti_min}) must be greater than 0.")

    # --- Am Ende alle Errors prüfen ---
    if errors:
        raise ConfigError("Configuration validation failed:\n" + "\n".join(errors))

    return cfg


# --- Test ---
if __name__ == "__main__":
    config = load_config()
    print("Configuration successfully loaded!")
    print(config)

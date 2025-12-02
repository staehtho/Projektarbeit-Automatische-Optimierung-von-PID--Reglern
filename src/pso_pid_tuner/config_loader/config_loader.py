# ──────────────────────────────────────────────────────────────────────────────
# Project:       PID Optimizer
# Module:        config_loader.py
# Description:   Loads and validates the YAML configuration for the PID Optimizer. This script
#                parses all system and PSO parameters, enforces type and value constraints,
#                maps string parameters to Enums, and returns a fully normalized configuration
#                dictionary. Invalid configurations raise a detailed ConfigError.
#
# Authors:       Florin Büchi, Thomas Stähli
# Created:       01.12.2025
# Modified:      01.12.2025
# Version:       1.0
#
# License:       ZHAW Zürcher Hochschule für angewandte Wissenschaften (or internal use only)
# ──────────────────────────────────────────────────────────────────────────────


import yaml
from pathlib import Path
import sys

from ..controlsys import AntiWindup, PerformanceIndex


class ConfigError(Exception):
    """Custom exception indicating invalid or inconsistent configuration input.

    This exception is raised when the configuration file cannot be loaded,
    contains invalid data types, violates logical constraints, or fails any of
    the validation rules applied during parsing.

    Typical causes include:
        - Missing required configuration sections.
        - Invalid numerical ranges (e.g., min ≥ max).
        - Unsupported string literals for enumerations.
        - Structural errors in ``config.yaml``.

    Attributes:
        message (str): Human-readable description of the configuration issue.
    """
    pass


def get_config_path() -> Path:
    """
    Resolve the filesystem path to the active ``config.yaml`` file.

    The function transparently supports both **script execution** and
    **frozen application mode** (e.g., PyInstaller-built binaries).
    In frozen mode, the configuration file is expected in the same directory
    as the executable. In script mode, the function locates the project root
    based on the current file position and returns ``./config/config.yaml``.

    Returns:
        Path: Absolute path to the ``config.yaml`` configuration file.

    Notes:
        - Frozen mode is detected via ``sys.frozen``.
        - Script mode uses the directory structure to locate the project root.
        - No validation of file existence is performed.
    """
    if getattr(sys, "frozen", False):
        # EXE-Modus → config.yaml liegt neben der EXE
        base_path = Path(sys.executable).parent
        return base_path / "config.yaml"
    else:
        # Script-Modus → zurück zur Projekt-Wurzel gehen
        # current file = main/config_loader/config_loader.py
        # parent -> config_loader
        # parent -> main/
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "config" / "config.yaml"


def load_config():
    """
    Load, validate, and normalize the PID Optimizer configuration file.

    This function reads the ``config.yaml`` file, performs extensive structural,
    type, and value validation on all configuration sections, and converts
    string-encoded parameters into internal enum representations. All detected
    validation issues are accumulated and reported collectively to provide
    actionable feedback to the user.

    The function ensures that required configuration sections exist, validates
    numeric ranges (e.g., controller bounds, simulation settings), checks
    logical relationships (e.g., min < max constraints), and prepares a fully
    normalized configuration dictionary for downstream processing.

    Returns:
        dict: A validated and normalized configuration dictionary containing:
            - ``system`` definitions (plant, simulation settings, constraints, enums)
            - ``pso`` settings (swarm size, iterations, parameter bounds)

    Raises:
        ConfigError:
            If the configuration file cannot be loaded or if any validation
            errors are detected. All issues are aggregated into a single error
            message.

    Notes:
        - ``anti_windup`` and ``performance_index`` fields are mapped to Enum
          instances and also stored as lowercase strings where required.
        - Numeric strings such as ``"1e-4"`` are automatically converted to floats.
        - This function performs only semantic validation; it does not verify
          whether the provided system dynamics are physically realizable.
    """
    config_path = get_config_path()

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

    mode = sim_time.get("mode")
    mode = mode.strip().lower()
    if mode not in ("fixed", "auto"):
        errors.append(
            f"'mode' must be either 'fixed' or 'auto', got '{mode}'."
        )
    sim_time["mode"] = mode

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
    else:
        match anti_windup:
            case "clamping":
                cfg["system"]["anti_windup"] = AntiWindup.CLAMPING
            case "conditional":
                cfg["system"]["anti_windup"] = AntiWindup.CONDITIONAL
        cfg["system"]["anti_windup_string"] = cfg["system"]["anti_windup"].name.lower()

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

    performance_index = system.get("performance_index")
    if performance_index not in ["IAE", "ISE", "ITAE", "ITSE"]:
        errors.append(f"'performance_index' should be 'ITAE' or 'IAE', got {performance_index}")
    else:
        match performance_index:
            case "IAE":
                cfg["system"]["performance_index"] = PerformanceIndex.IAE
            case "ISE":
                cfg["system"]["performance_index"] = PerformanceIndex.ISE
            case "ITAE":
                cfg["system"]["performance_index"] = PerformanceIndex.ITAE
            case "ITSE":
                cfg["system"]["performance_index"] = PerformanceIndex.ITSE

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

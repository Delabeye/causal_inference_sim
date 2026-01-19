# utilities/config.py
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml"):
    """
    Load and validate a YAML configuration file with UTF-8 encoding.
    This function reads a YAML configuration file and performs comprehensive validation
    to ensure the file exists, is not empty, and contains valid YAML dictionary structure.
    Args:
        config_path (str, optional): Path to the YAML configuration file. 
            Defaults to "config.yaml".
    Returns:
        dict: A dictionary containing the parsed YAML configuration.
    Raises:
        FileNotFoundError: If the configuration file does not exist at the specified path.
        ValueError: If the configuration file is empty or contains only comments.
        TypeError: If the configuration file does not represent a YAML dictionary.
    Example:
        >>> config = load_config("config.yaml")
        Configuration loaded from config.yaml
        >>> isinstance(config, dict)
        True
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found "
            f"(current directory = {Path.cwd()})"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        # File is empty or contains only comments
        raise ValueError(f"Configuration file '{config_path}' is empty.")

    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration file '{config_path}' does not represent a YAML dictionary."
        )

    print(f"Configuration loaded from {config_path}")
    return config
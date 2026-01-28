# utilities/config.py
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml"):
    """
    Loads and validates a YAML configuration file.

    This utility handles UTF-8 encoded YAML files and performs several 
    integrity checks to ensure the simulation starts with valid parameters. 
    It verifies file existence, confirms the content is not empty, and 
    validates that the top-level structure is a dictionary.

    Args:
        config_path (str): Path to the .yaml configuration file. 
                       Defaults to "config.yaml".

    Returns:
        dict: The parsed configuration parameters.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or contains only comments.
        TypeError: If the YAML content is not formatted as a dictionary.
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
        # Empty file or just comments
        raise ValueError(f"Configuration file '{config_path}' is empty.")

    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration file '{config_path}' is not a YAML dictionnary."
        )

    print(f"Configuration loaded from {config_path}")
    return config


def save_config(config: dict, config_path: str) -> None:
    """
    Serializes a configuration dictionary to a YAML file.

    Automates directory creation if the target path does not exist and 
    exports the dictionary using UTF-8 encoding. It preserves the 
    original key order and ensures non-ASCII characters are handled 
    correctly via unicode support.

    Args:
        config (dict): The configuration data to be saved.
        config_path (str): The destination file path.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


# utilities/config.py

import yaml


def load_config(config_path: str = "config.yaml"):
    """
    Charge le fichier de configuration YAML (encodage UTF-8).
    """
    try:
        # ✅ on force l'encodage en UTF-8 pour éviter les erreurs Windows (cp1252)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print(f"Configuration chargée depuis {config_path}")
        return config

    except FileNotFoundError:
        print(f"ERREUR : Fichier de configuration '{config_path}' non trouvé.")
        return None

    except UnicodeDecodeError as e:
        print(f"ERREUR : Problème d'encodage en lisant '{config_path}' : {e}")
        return None

    except yaml.YAMLError as e:
        print(f"ERREUR : Erreur lors de l'analyse du YAML : {e}")
        return None

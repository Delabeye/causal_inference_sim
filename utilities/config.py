import yaml

def load_config(config_path='config.yaml'):
    """
    Charge le fichier de configuration YAML.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration chargée depuis {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERREUR : Fichier de configuration '{config_path}' non trouvé.")
        return None
    except yaml.YAMLError as e:
        print(f"ERREUR : Erreur lors de l'analyse du YAML : {e}")
        return None
    
    
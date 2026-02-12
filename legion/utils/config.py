import yaml
from pathlib import Path


class ConfigUtil:
    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir).resolve(True)

    def load(self, name: str):
        # Extension is optional, append it if not present
        if name.endswith(".yaml"):
            filename = name
        else:
            filename = f"{name}.yaml"

        # Load from file
        file_path = (self.config_dir / filename).resolve(True)
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Resolve parent configs
        if "extends" in cfg:
            parent = self.load(cfg.pop("extends"))
            override_dict(parent, cfg)
            return parent
        else:
            return cfg


def override_dict(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            override_dict(base[k], v)
        else:
            base[k] = v

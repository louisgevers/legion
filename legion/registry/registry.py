# Different registries
EMBODIMENTS: dict[str, type] = {}
PHYSICS: dict[str, type] = {}
ACTUATORS: dict[str, type] = {}
SIGNALS: dict[str, type] = {}
REWARDS: dict[str, type] = {}
OBSERVATIONS: dict[str, type] = {}
TERMINATIONS: dict[str, type] = {}


def register(registry: dict[str, type], name: str):
    def decorator(cls):
        registry[name] = cls
        return cls

    return decorator


def build(cfg: dict, registry: dict[str, type], **extra_kwargs):
    cfg = dict(cfg)  # Copy config
    cls_name = cfg.pop("type")
    cls = registry[cls_name]
    return cls(**cfg, **extra_kwargs)

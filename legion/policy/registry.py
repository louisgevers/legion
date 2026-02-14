POLICIES = {}


def register_policy(name):
    def decorator(cls):
        POLICIES[name] = cls
        cls._policy_name = name
        return cls

    return decorator

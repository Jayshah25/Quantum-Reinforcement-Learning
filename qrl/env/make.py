
from qrl.env.registry import ENV_REGISTRY

def make(env_name: str):
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Environment '{env_name}' not found in registry.")
    return ENV_REGISTRY[env_name]()

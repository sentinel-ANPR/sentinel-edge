import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent

def resolve_model_path(env_var: str, default_relative_path: str) -> str:
    raw_value = os.getenv(env_var, default_relative_path)
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return str(candidate)
    return str((APP_ROOT / candidate).resolve())

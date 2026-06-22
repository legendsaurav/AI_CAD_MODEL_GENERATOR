"""
utils/config.py - Configuration Manager
BUG FIX: The singleton used a relative path which breaks when scripts are run from
any directory other than the repo root.  We now resolve the config path relative
to THIS file so it always works regardless of CWD.
"""
import os
import yaml

# Absolute path to the default config so it works from any working directory
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


class ConfigManager:
    """Loads and manages the global configuration for the Geometry Engine."""

    _instance = None
    _config: dict = {}

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            path = config_path or _DEFAULT_CONFIG
            cls._instance._load_config(os.path.normpath(path))
        return cls._instance

    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}

    @classmethod
    def reset(cls):
        """Reset singleton – useful for unit-testing with different configs."""
        cls._instance = None
        cls._config = {}

    @classmethod
    def get(cls, key_path: str, default=None):
        """
        Retrieves a value using dot-notation.
        Example: ConfigManager.get("heads.confidence_threshold")
        """
        if not cls._instance:
            cls()
        keys = key_path.split(".")
        val = cls._instance._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    @classmethod
    def get_all(cls) -> dict:
        if not cls._instance:
            cls()
        return cls._instance._config

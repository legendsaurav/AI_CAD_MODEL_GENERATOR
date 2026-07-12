import importlib
from typing import Dict, Any
from heads.base import GeometryHeadBase

class HeadPluginLoader:
    """
    Dynamically loads geometry heads based on the global configuration.
    This prevents hardcoding prediction heads into the architecture.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_heads = config.get("heads", {}).get("enabled", [])
        self.head_instances: Dict[str, GeometryHeadBase] = {}
        self._load_plugins()

    def _load_plugins(self):
        """Discovers and instantiates enabled heads."""
        for head_name in self.enabled_heads:
            module_name = f"heads.{head_name}"
            try:
                # Dynamically import the module
                module = importlib.import_module(module_name)
                
                # Find the class that inherits from GeometryHeadBase
                head_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, GeometryHeadBase) and attr is not GeometryHeadBase:
                        head_class = attr
                        break
                
                if head_class is None:
                    print(f"⚠️ Warning: Could not find a valid GeometryHeadBase class in {module_name}")
                    continue
                
                # Instantiate and store
                instance = head_class(self.config.get("heads", {}))
                self.head_instances[head_name] = instance
                print(f"🔌 Successfully loaded head plugin: {head_name}")
                
            except ImportError as e:
                print(f"⚠️ Warning: Failed to load plugin {head_name} - {e}")

    def get_all_heads(self) -> Dict[str, GeometryHeadBase]:
        return self.head_instances

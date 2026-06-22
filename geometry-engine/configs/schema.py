"""
Config Validation Schema for Geometry Engine
==============================================
Pydantic models that validate config.yaml at load time.
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator
import yaml
import os


class HooksConfig(BaseModel):
    """Configuration for DiT forward hook registration."""
    double_block_indices: List[int] = Field(default=[0, 4, 8, 12])
    single_block_indices: List[int] = Field(default=[0, 4, 8])

    @field_validator("double_block_indices", "single_block_indices")
    @classmethod
    def indices_non_negative(cls, v: List[int]) -> List[int]:
        if any(i < 0 for i in v):
            raise ValueError("Hook indices must be non-negative")
        return v


class ProbingConfig(BaseModel):
    """Configuration for representation probing analysis."""
    pca_components: int = Field(default=50, ge=2, le=1000)
    correlation_method: Literal["pearson", "spearman"] = "pearson"
    temporal_window: int = Field(default=5, ge=1)
    auto_select_layers: bool = True
    probe_epochs: int = Field(default=100, ge=1)


class HeadConfig(BaseModel):
    """Configuration for a single geometry head."""
    name: str
    enabled: bool = True
    hidden_dim: int = Field(default=512, ge=32)
    num_classes: int = Field(default=8, ge=2)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)


class HeadsConfig(BaseModel):
    """Configuration for all geometry heads."""
    enabled_heads: List[HeadConfig] = Field(default_factory=list)
    uncertainty_samples: int = Field(default=20, ge=1)


class PrimitiveConfig(BaseModel):
    """Configuration for primitive fitting."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_proposals: int = Field(default=5, ge=1)
    fitting_method: Literal["ransac", "least_squares", "levenberg_marquardt", "differentiable"] = "ransac"
    ransac_max_trials: int = Field(default=1000, ge=10)
    ransac_residual_threshold: float = Field(default=0.01, gt=0.0)


class TrainingConfig(BaseModel):
    """Configuration for head training."""
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=100, ge=1)
    loss_weights: Dict[str, float] = Field(default_factory=lambda: {
        "segmentation": 1.0, "contrastive": 0.5, "topology": 0.3
    })
    checkpoint_dir: str = "checkpoints"
    training_mode: Literal["supervised", "contrastive", "masked_prediction", "multi_task"] = "supervised"


class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty estimation."""
    method: Literal["mc_dropout", "ensemble", "temperature"] = "mc_dropout"
    num_samples: int = Field(default=20, ge=1)
    temperature: float = Field(default=1.0, gt=0.0)


class GGLConfig(BaseModel):
    """Configuration for GGL output."""
    max_nodes: int = Field(default=50, ge=1)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class LoggingConfig(BaseModel):
    """Configuration for structured logging."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "colored"] = "colored"
    log_dir: Optional[str] = "logs"


class RefinementConfig(BaseModel):
    """Configuration for the iterative refinement loop."""
    max_iterations: int = Field(default=5, ge=1)
    chamfer_threshold: float = Field(default=0.5, gt=0.0)
    hausdorff_threshold: float = Field(default=2.0, gt=0.0)
    normal_consistency_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class GeometryEngineConfig(BaseModel):
    """
    Top-level configuration for the Geometry Engine.
    Validates all sub-configurations at load time.
    """
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    probing: ProbingConfig = Field(default_factory=ProbingConfig)
    heads: HeadsConfig = Field(default_factory=HeadsConfig)
    primitive: PrimitiveConfig = Field(default_factory=PrimitiveConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    ggl: GGLConfig = Field(default_factory=GGLConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    @classmethod
    def load_from_yaml(cls, path: str) -> "GeometryEngineConfig":
        """Load and validate configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def default(cls) -> "GeometryEngineConfig":
        """Return a default configuration with all defaults."""
        return cls()

"""
Experiment Tracker for Geometry Engine
========================================
Logs metrics, saves checkpoints, and tracks hyperparameters
using JSON-lines format for portability.
"""
import json
import os
import time
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks experiments: metrics, configs, checkpoints.

    Storage layout::

        base_dir/
          experiment_name_YYYYMMDD_HHMMSS/
            config.json
            metrics.jsonl
            checkpoints/
              epoch_001.pt
              best.pt
    """

    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = experiment_name
        self.run_dir = os.path.join(base_dir, f"{experiment_name}_{ts}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._config_path = os.path.join(self.run_dir, "config.json")
        self._best_metric: Optional[float] = None
        self._best_ckpt: Optional[str] = None
        self._all_metrics: List[Dict[str, Any]] = []
        self._start_time = time.time()

        logger.info(f"Experiment '{experiment_name}' -> {self.run_dir}")

    # ── Config ────────────────────────────────────────────────

    def log_config(self, config: Dict[str, Any]) -> None:
        """Save the experiment configuration snapshot."""
        meta = {
            "experiment_name": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_hash": self._get_git_hash(),
            "config": config,
        }
        with open(self._config_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Config saved")

    # ── Metrics ───────────────────────────────────────────────

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a single metric value at a given step."""
        entry = {"step": step, "name": name, "value": value,
                 "timestamp": datetime.now(timezone.utc).isoformat()}
        self._all_metrics.append(entry)
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    # ── Checkpoints ───────────────────────────────────────────

    def save_checkpoint(
        self, model, optimizer, epoch: int, metrics: Dict[str, float]
    ) -> str:
        """Save a training checkpoint. Tracks best by val_loss."""
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available; skipping checkpoint save")
            return ""

        path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

        val_loss = metrics.get("val_loss", metrics.get("loss", float("inf")))
        if self._best_metric is None or val_loss < self._best_metric:
            self._best_metric = val_loss
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            torch.save(torch.load(path, weights_only=False), best_path)
            self._best_ckpt = best_path
            logger.info(f"New best checkpoint (val_loss={val_loss:.4f})")

        return path

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint file."""
        import torch
        return torch.load(path, weights_only=False)

    def get_best_checkpoint(self) -> Optional[str]:
        """Return path to the best checkpoint by validation loss."""
        return self._best_ckpt

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the experiment."""
        elapsed = time.time() - self._start_time
        metric_names = set(m["name"] for m in self._all_metrics)
        bests = {}
        for name in metric_names:
            vals = [m["value"] for m in self._all_metrics if m["name"] == name]
            bests[name] = {"min": min(vals), "max": max(vals), "last": vals[-1]}
        return {
            "experiment_name": self.name,
            "run_dir": self.run_dir,
            "elapsed_seconds": round(elapsed, 1),
            "total_metric_entries": len(self._all_metrics),
            "best_values": bests,
            "best_checkpoint": self._best_ckpt,
        }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _get_git_hash() -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

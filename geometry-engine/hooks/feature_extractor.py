"""
hooks/feature_extractor.py
===========================
CRITICAL BUG FIX:
  register_forward_pre_hook without `with_kwargs=True` passes only
  (module, args) to the hook – keyword arguments like `timestep=0.5`
  are silently dropped.  This meant `self.current_timestep` was never
  set, so `_should_capture()` always returned False and nothing was saved.

FIX: Try to register with `with_kwargs=True` (PyTorch ≥ 2.0).
     If that flag is not supported (PyTorch < 2.0), fall back to a
     2-argument compat hook that reads the timestep from positional args.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, cast


class DiTFeatureExtractor:
    """
    Registers forward hooks on a Hunyuan3D DiT to capture
    intermediate hidden states during inference.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List = []
        self.features: Dict[str, Any] = {
            "double_blocks": {},
            "single_blocks": {},
            "final_layer": None,
        }
        self.target_timesteps: List[float] = []
        self.current_timestep: Optional[float] = None
        self.enabled: bool = True

    # ── Public API ────────────────────────────────────────────────────────

    def set_target_timesteps(self, timesteps: List[float]):
        """Set diffusion timesteps at which to capture features (e.g. [0.1, 0.5, 0.9])."""
        self.target_timesteps = list(timesteps)

    def register_hooks(
        self,
        double_indices: Optional[List[int]] = None,
        single_indices: Optional[List[int]] = None,
    ):
        """Register hooks on specified layer indices of double_blocks and single_blocks."""
        self.clear_hooks()

        # ── Pre-hook: capture current timestep from the model's forward call ──
        # PyTorch ≥ 2.0 supports with_kwargs=True to receive keyword arguments.
        # Older versions only pass (module, args).
        try:
            handle = self.model.register_forward_pre_hook(
                self._model_pre_hook_with_kwargs, with_kwargs=True
            )
        except TypeError:
            # PyTorch < 2.0 fallback – reads timestep from positional args only
            handle = self.model.register_forward_pre_hook(
                self._model_pre_hook_positional
            )
        self.hooks.append(handle)

        # ── Double-stream block hooks ─────────────────────────────────────────
        if hasattr(self.model, "double_blocks") and double_indices is not None:
            double_blocks = cast(nn.ModuleList, self.model.double_blocks)
            for i in double_indices:
                if i < len(double_blocks):
                    h = double_blocks[i].register_forward_hook(
                        self._double_block_hook(i)
                    )
                    self.hooks.append(h)

        # ── Single-stream block hooks ─────────────────────────────────────────
        if hasattr(self.model, "single_blocks") and single_indices is not None:
            single_blocks = cast(nn.ModuleList, self.model.single_blocks)
            for i in single_indices:
                if i < len(single_blocks):
                    h = single_blocks[i].register_forward_hook(
                        self._single_block_hook(i)
                    )
                    self.hooks.append(h)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear_features(self):
        self.features = {
            "double_blocks": {},
            "single_blocks": {},
            "final_layer": None,
        }

    # ── Pre-hook implementations ──────────────────────────────────────────

    def _model_pre_hook_with_kwargs(self, module, args, kwargs):
        """PyTorch ≥ 2.0: kwargs are available."""
        t = kwargs.get("timestep", None)
        if t is None and len(args) > 1:
            t = args[1]
        self._set_timestep(t)

    def _model_pre_hook_positional(self, module, args):
        """PyTorch < 2.0 fallback: no kwargs, read from positional args."""
        t = args[1] if len(args) > 1 else None
        self._set_timestep(t)

    def _set_timestep(self, t):
        if t is None:
            return
        if isinstance(t, torch.Tensor):
            t = t.item()
        self.current_timestep = float(t)

    # ── Block hook factories ──────────────────────────────────────────────

    def _double_block_hook(self, idx: int):
        def hook(module, input, output):
            if not self._should_capture():
                return
            t = self.current_timestep
            self.features["double_blocks"].setdefault(t, {})

            img_t = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
            txt_t = output[1].detach().cpu() if (isinstance(output, tuple) and len(output) > 1) else None

            self.features["double_blocks"][t][idx] = {"img": img_t, "txt": txt_t}
        return hook

    def _single_block_hook(self, idx: int):
        def hook(module, input, output):
            if not self._should_capture():
                return
            t = self.current_timestep
            self.features["single_blocks"].setdefault(t, {})

            out = output[0] if isinstance(output, tuple) else output
            self.features["single_blocks"][t][idx] = out.detach().cpu()
        return hook

    # ── Helper ───────────────────────────────────────────────────────────

    def _should_capture(self) -> bool:
        if not self.enabled:
            return False
        if not self.target_timesteps:
            return True                              # capture every step if no filter set
        return self.current_timestep in self.target_timesteps

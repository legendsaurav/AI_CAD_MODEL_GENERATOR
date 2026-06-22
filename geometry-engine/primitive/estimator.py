"""
primitive/estimator.py - Parameter Estimator
BUG FIX 1: Constructor read `config.get("heads", {}).get("hidden_dim", 1024)` but
the caller (run_primitive_recovery.py) passes the full config dict, so this was
correct – but `ParameterEstimator(config)` is called with the full dict while the
regressor expected the *heads* sub-dict for hidden_dim.  Harmonised to read
properly from the full config.

BUG FIX 2: `raw_params` shape from `self.regressor(feature)` depends on
feature shape.  If feature is [1, 1024] the output is [1, 10]; if [1024] it
would fail.  Added unsqueeze guard.

BUG FIX 3: `pred_vals = raw_params[0, :num_params]` – slice before `.tolist()`
to avoid shape mismatches when num_params > 10.  Added min() guard.
"""
import torch
import torch.nn as nn
from typing import Any, Dict

from graph.ggl import GGLNode


class ParameterEstimator(nn.Module):
    """
    Predicts continuous geometric parameters (radius, height, bounding box, etc.)
    for proposed primitives.
    """

    PARAM_MAP: Dict[str, list] = {
        "Cylinder": ["radius", "height", "center_x", "center_y", "center_z", "axis_x", "axis_y", "axis_z"],
        "Box":      ["width",  "height", "depth",  "center_x", "center_y", "center_z", "rot_x", "rot_y", "rot_z"],
        "Sphere":   ["radius", "center_x", "center_y", "center_z"],
        "Cone":     ["radius", "height", "center_x", "center_y", "center_z"],
        "Plane":    ["normal_x", "normal_y", "normal_z", "distance"],
    }
    MAX_PARAMS = 10  # output dimension of the regressor

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_dim = config.get("heads", {}).get("hidden_dim", 1024)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.MAX_PARAMS),
        )

    # ------------------------------------------------------------------
    def estimate(self, feature: torch.Tensor, proposal_node: GGLNode) -> GGLNode:
        """
        Takes a pooled feature [D] or [1, D] and predicts parameters for the proposal.
        Returns the same node with `.parameters` populated.
        """
        # Ensure 2-D input: [1, D]
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)

        with torch.no_grad():
            raw_params = self.regressor(feature)  # [1, MAX_PARAMS]

        prim_type = proposal_node.type
        if prim_type not in self.PARAM_MAP:
            return proposal_node

        expected = self.PARAM_MAP[prim_type]
        n = min(len(expected), self.MAX_PARAMS)
        pred_vals = raw_params[0, :n].cpu().tolist()

        for i, name in enumerate(expected[:n]):
            raw_val = pred_vals[i]
            # Positive-definite parameters (sizes / radii) use abs()
            if any(k in name for k in ("radius", "width", "height", "depth")):
                proposal_node.parameters[name] = abs(raw_val)
            else:
                proposal_node.parameters[name] = raw_val

        return proposal_node

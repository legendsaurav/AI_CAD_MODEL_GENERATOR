"""
desktop-agent/solidworks_executor.py — SolidWorks API Executor
================================================================
Executes CAL actions in SolidWorks via COM/Win32 automation.

Requires SolidWorks to be installed on the host machine.
Uses pythoncom and win32com for COM automation.

NOTE: This runs on the HOST machine (not in Docker) because
SolidWorks requires GUI access.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from executor import BaseExecutor

logger = logging.getLogger("desktop_agent.solidworks_executor")


class SolidWorksExecutor(BaseExecutor):
    """
    Executes CAL actions in SolidWorks via COM automation.

    Connection flow:
      1. Get running SolidWorks instance via COM
      2. Create a new part document
      3. Execute CAL actions as SolidWorks API calls
      4. Export STL/STEP for verification
    """

    @property
    def target_system(self) -> str:
        return "SolidWorks"

    def __init__(self) -> None:
        self._app = None
        self._model = None
        self._connected = False
        self._feature_count = 0

    def connect(self) -> bool:
        """Connect to SolidWorks via COM automation."""
        try:
            import win32com.client
            self._app = win32com.client.Dispatch("SldWorks.Application")
            self._app.Visible = True
            # Create new part
            self._model = self._app.NewDocument(
                self._app.GetUserPreferenceStringValue(24),  # swDefaultTemplatePart
                0, 0, 0,
            )
            self._connected = True
            logger.info("Connected to SolidWorks.")
            return True
        except ImportError:
            logger.error("win32com not available. Install pywin32.")
            return False
        except Exception as e:
            logger.error("Failed to connect to SolidWorks: %s", e)
            return False

    def execute_action(
        self, action_type: str, action_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a CAL action in SolidWorks."""
        if not self._connected:
            return {"success": False, "feature_id": None, "error": "Not connected"}

        t0 = time.time()
        try:
            dispatch = {
                "create_sketch": self._create_sketch,
                "draw_circle": self._draw_circle,
                "draw_rectangle": self._draw_rectangle,
                "extrude": self._extrude,
                "revolve": self._revolve,
                "fillet": self._fillet,
                "chamfer": self._chamfer,
            }
            handler = dispatch.get(action_type)
            if handler is None:
                return {
                    "success": False,
                    "feature_id": None,
                    "error": f"Unsupported action: {action_type}",
                }
            result = handler(action_id, params)
            elapsed = (time.time() - t0) * 1000
            result["execution_time_ms"] = elapsed
            return result
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.error(
                "Action %s (%s) failed after %.0fms: %s",
                action_id, action_type, elapsed, e,
            )
            return {"success": False, "feature_id": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _create_sketch(self, action_id: str, params: Dict) -> Dict:
        """Create a sketch on the specified plane."""
        plane_map = {
            "XY": "Front Plane",
            "XZ": "Top Plane",
            "YZ": "Right Plane",
        }
        plane_name = plane_map.get(params.get("plane", "XY"), "Front Plane")

        if self._model:
            # Select plane and insert sketch
            self._model.Extension.SelectByID2(
                plane_name, "PLANE", 0, 0, 0, False, 0, None, 0
            )
            self._model.SketchManager.InsertSketch(True)
            logger.info("Created sketch '%s' on %s", action_id, plane_name)

        return {"success": True, "feature_id": action_id, "error": None}

    def _draw_circle(self, action_id: str, params: Dict) -> Dict:
        """Draw a circle in the active sketch."""
        cx, cy = params.get("center", [0, 0])
        r = params.get("radius", 1.0)
        # SolidWorks uses meters internally
        cx_m, cy_m, r_m = cx / 1000, cy / 1000, r / 1000

        if self._model:
            self._model.SketchManager.CreateCircle(cx_m, cy_m, 0, cx_m + r_m, cy_m, 0)
            logger.info("Drew circle r=%.2fmm at (%.2f, %.2f)", r, cx, cy)

        return {"success": True, "feature_id": action_id, "error": None}

    def _draw_rectangle(self, action_id: str, params: Dict) -> Dict:
        """Draw a rectangle in the active sketch."""
        cx, cy = params.get("center", [0, 0])
        w = params.get("width", 1.0) / 1000
        h = params.get("height", 1.0) / 1000
        x0, y0 = (cx / 1000) - w / 2, (cy / 1000) - h / 2
        x1, y1 = x0 + w, y0 + h

        if self._model:
            self._model.SketchManager.CreateCornerRectangle(x0, y0, 0, x1, y1, 0)
            logger.info("Drew rectangle at center (%.2f, %.2f)", cx, cy)

        return {"success": True, "feature_id": action_id, "error": None}

    def _extrude(self, action_id: str, params: Dict) -> Dict:
        """Extrude the active sketch."""
        depth = params.get("depth", 10.0) / 1000  # mm → meters
        is_cut = params.get("is_cut", False)

        if self._model:
            # Close sketch first
            self._model.SketchManager.InsertSketch(True)
            mgr = self._model.FeatureManager
            if is_cut:
                mgr.FeatureCut4(
                    True, 0, 0, 0, 0, depth, depth,
                    False, False, False, False, 0, 0,
                    False, False, False, False, False,
                    True, True, True, True, False, 0, 0, False, False
                )
            else:
                mgr.FeatureExtrusion3(
                    True, False, False, 0, 0, depth, depth,
                    False, False, False, False, 0, 0,
                    False, False, False, False, True, False, False
                )
            self._feature_count += 1
            logger.info("Extruded %.2fmm (%s)", depth * 1000, "cut" if is_cut else "boss")

        return {"success": True, "feature_id": action_id, "error": None}

    def _revolve(self, action_id: str, params: Dict) -> Dict:
        """Revolve the active sketch."""
        if self._model:
            self._model.SketchManager.InsertSketch(True)
            self._model.FeatureManager.FeatureRevolve2(
                True, True, False, False, False, False,
                0, 0, 6.28318, 0, False, False, 0, 0
            )
            self._feature_count += 1
            logger.info("Revolved sketch 360°")

        return {"success": True, "feature_id": action_id, "error": None}

    def _fillet(self, action_id: str, params: Dict) -> Dict:
        """Apply fillet to selected edges."""
        radius = params.get("radius", 1.0) / 1000
        if self._model:
            self._model.FeatureManager.FeatureFillet3(
                194, radius, 0, 0, 0, 0, 0, None, None, None, None, None, None, None
            )
            logger.info("Applied fillet r=%.2fmm", radius * 1000)
        return {"success": True, "feature_id": action_id, "error": None}

    def _chamfer(self, action_id: str, params: Dict) -> Dict:
        """Apply chamfer to selected edges."""
        distance = params.get("distance", 1.0) / 1000
        if self._model:
            self._model.FeatureManager.InsertFeatureChamfer(
                4, 1, distance, 0.7854, 0, 0, 0, 0
            )
            logger.info("Applied chamfer d=%.2fmm", distance * 1000)
        return {"success": True, "feature_id": action_id, "error": None}

    # ------------------------------------------------------------------
    # Export and utility
    # ------------------------------------------------------------------

    def export_mesh(self, output_path: str, format: str = "stl") -> bool:
        """Export the SolidWorks model as STL."""
        try:
            if self._model:
                # NOTE: `format` is currently ignored — SaveAs derives the
                # export type from output_path's extension. Preserved as a
                # known limitation rather than silently computing an unused var.
                self._model.Extension.SaveAs(
                    output_path, 0, 2, None, None, None
                )
                logger.info("Exported to %s", output_path)
                return True
        except Exception as e:
            logger.error("Export failed: %s", e)
        return False

    def take_screenshot(self, output_path: str) -> bool:
        """Capture the SolidWorks viewport."""
        try:
            if self._model:
                self._model.SaveBMP(output_path, 1920, 1080)
                logger.info("Screenshot saved: %s", output_path)
                return True
        except Exception as e:
            logger.warning("Screenshot failed: %s", e)
        return False

    def undo(self) -> bool:
        """Undo the last SolidWorks operation."""
        try:
            if self._model:
                self._model.EditUndo2(1)
                return True
        except Exception:
            pass
        return False

    def disconnect(self) -> None:
        """Close SolidWorks document (leave app running)."""
        try:
            if self._app and self._model:
                self._app.CloseDoc(self._model.GetTitle())
        except Exception:
            pass
        self._model = None
        self._connected = False
        logger.info("Disconnected from SolidWorks.")

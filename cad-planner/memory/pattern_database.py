"""
memory/pattern_database.py — Engineering Construction Pattern Database
========================================================================
Stores reusable construction patterns derived from standard mechanical
engineering practice. Each pattern encodes a proven construction sequence
for a common engineering feature.

Patterns can be loaded from:
  1. Built-in hardcoded patterns (always available)
  2. JSON pattern files (extensible)
  3. Learned patterns from successful past sessions (future)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("cad_planner.memory.pattern_database")


class PatternDatabase:
    """
    Database of engineering construction patterns.

    Each pattern maps a feature name to an ordered list of CAD operations
    that construct that feature. Patterns encode expert knowledge about
    the most robust, editable, and manufacturable way to create features.
    """

    def __init__(self, pattern_dir: Optional[str] = None) -> None:
        # Built-in patterns from engineering best practice
        self.patterns: Dict[str, List[str]] = {
            # Hole features
            "standard_hole": [
                "create_sketch", "extrude_cut",
            ],
            "counterbore_hole": [
                "create_sketch", "extrude_cut",
                "extrude_cut_counterbore", "chamfer",
            ],
            "countersink_hole": [
                "create_sketch", "extrude_cut",
                "chamfer_countersink",
            ],
            "tapped_hole": [
                "create_sketch", "extrude_cut",
                "thread_cosmetic",
            ],
            # Shaft features
            "bearing_seat": [
                "create_sketch", "extrude",
                "fillet", "pattern",
            ],
            "keyway": [
                "create_sketch", "extrude_cut",
                "fillet",
            ],
            "spline": [
                "create_sketch", "extrude",
                "pattern_circular",
            ],
            # Plate/body features
            "flange": [
                "create_sketch", "revolve",
                "create_sketch", "extrude_cut",
                "pattern", "fillet",
            ],
            "boss": [
                "create_sketch", "extrude",
                "fillet",
            ],
            "pocket": [
                "create_sketch", "extrude_cut",
                "fillet",
            ],
            "slot": [
                "create_sketch", "extrude_cut",
            ],
            "rib": [
                "create_sketch", "extrude",
                "draft",
            ],
            # Edge features
            "chamfer_edge": [
                "chamfer",
            ],
            "fillet_edge": [
                "fillet",
            ],
            # Shell/thin-wall
            "shell": [
                "shell",
            ],
            # Pattern features
            "bolt_pattern": [
                "create_sketch", "extrude_cut",
                "pattern_circular",
            ],
            "linear_pattern": [
                "create_sketch", "extrude",
                "pattern_linear",
            ],
        }

        # Load additional patterns from directory
        if pattern_dir and os.path.isdir(pattern_dir):
            self._load_from_directory(pattern_dir)

    def get_pattern(self, name: str) -> List[str]:
        """Get a construction pattern by name."""
        return self.patterns.get(name, [])

    def get_all_patterns(self) -> Dict[str, List[str]]:
        """Return all registered patterns."""
        return dict(self.patterns)

    def register_pattern(self, name: str, sequence: List[str]) -> None:
        """Register a new pattern (or overwrite existing)."""
        self.patterns[name] = sequence
        logger.info("Registered pattern '%s': %s", name, sequence)

    def pattern_names(self) -> List[str]:
        """Return all pattern names."""
        return list(self.patterns.keys())

    def _load_from_directory(self, pattern_dir: str) -> None:
        """Load pattern definitions from JSON files in a directory."""
        loaded = 0
        for filename in os.listdir(pattern_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(pattern_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for name, sequence in data.items():
                        if isinstance(sequence, list):
                            self.patterns[name] = sequence
                            loaded += 1
            except Exception as e:
                logger.warning("Failed to load pattern file %s: %s", filepath, e)

        if loaded > 0:
            logger.info("Loaded %d patterns from %s", loaded, pattern_dir)

"""
desktop-agent/state_machine.py — Execution State Machine
==========================================================
Manages the state of CAD action execution to enable pause, resume,
rollback, and step-by-step operation.
"""
import logging
from typing import List, Optional
from enum import Enum

logger = logging.getLogger("desktop_agent.state_machine")


class ExecutionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActionState:
    """Tracks the execution state of a single CAL action."""

    def __init__(self, action_id: str, action_type: str):
        self.action_id = action_id
        self.action_type = action_type
        self.state: ExecutionState = ExecutionState.IDLE
        self.error: Optional[str] = None
        self.cad_feature_id: Optional[str] = None
        self.execution_time_ms: float = 0.0


class ExecutionStateMachine:
    """
    Manages execution state across the full CAL action sequence.

    Supports:
        - Sequential action execution
        - Pause/resume
        - Rollback (undo last N actions)
        - Step-by-step execution
        - State persistence for crash recovery
    """

    def __init__(self):
        self.state: ExecutionState = ExecutionState.IDLE
        self.action_states: List[ActionState] = []
        self.current_index: int = 0

    def load_actions(self, action_ids: List[tuple]):
        """Load action IDs and types from a CAL document."""
        self.action_states = [
            ActionState(action_id=aid, action_type=atype)
            for aid, atype in action_ids
        ]
        self.current_index = 0
        self.state = ExecutionState.IDLE
        logger.info(f"Loaded {len(self.action_states)} actions into state machine.")

    @property
    def current_action(self) -> Optional[ActionState]:
        if 0 <= self.current_index < len(self.action_states):
            return self.action_states[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.action_states)

    def advance(self):
        """Mark current action as complete and advance to next."""
        if self.current_action:
            self.current_action.state = ExecutionState.COMPLETED
            self.current_index += 1

        if self.is_complete:
            self.state = ExecutionState.COMPLETED
            logger.info("All actions completed.")

    def fail_current(self, error: str):
        """Mark current action as failed."""
        if self.current_action:
            self.current_action.state = ExecutionState.FAILED
            self.current_action.error = error
            self.state = ExecutionState.FAILED
            logger.error(f"Action {self.current_action.action_id} failed: {error}")

    def pause(self):
        """Pause execution."""
        self.state = ExecutionState.PAUSED
        logger.info(f"Execution paused at action {self.current_index}.")

    def resume(self):
        """Resume execution."""
        self.state = ExecutionState.RUNNING
        logger.info(f"Execution resumed at action {self.current_index}.")

    def rollback(self, steps: int = 1):
        """Roll back the last N completed actions."""
        for _ in range(steps):
            if self.current_index > 0:
                self.current_index -= 1
                self.action_states[self.current_index].state = ExecutionState.ROLLED_BACK
                logger.info(f"Rolled back action {self.action_states[self.current_index].action_id}")
        self.state = ExecutionState.PAUSED

    def get_summary(self) -> dict:
        """Get a summary of execution progress."""
        completed = sum(1 for a in self.action_states if a.state == ExecutionState.COMPLETED)
        failed = sum(1 for a in self.action_states if a.state == ExecutionState.FAILED)
        return {
            "state": self.state.value,
            "total": len(self.action_states),
            "completed": completed,
            "failed": failed,
            "remaining": len(self.action_states) - self.current_index,
            "current_index": self.current_index,
        }

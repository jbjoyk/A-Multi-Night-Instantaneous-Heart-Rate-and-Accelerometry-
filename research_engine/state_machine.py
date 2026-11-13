"""
research_engine/state_machine.py: Experiment State Machine & Lifecycle Governance.

Enforces strict sequential state transitions:
  PROPOSED -> GOVERNED -> DESIGNED -> REVIEWED -> APPROVED -> IMPLEMENTING -> RUNNING -> EVALUATING -> COMPLETED -> DECISION -> MEMORY

Terminal / Exception outcomes:
  FAILED, BLOCKED, DISCARDED, NEEDS_REVIEW

Strict Rules:
  1. Illegal transitions (e.g. PROPOSED -> KEPT, RUNNING -> KEPT, PROPOSED -> RUNNING) raise IllegalStateTransitionError.
  2. Execution modes (REAL, SIMULATION, DRY_RUN) are decoupled from lifecycle statuses.
  3. In-place conversion of SIMULATION or DRY_RUN into REAL is strictly prohibited.
"""

from enum import Enum
from typing import Dict, Set, Optional, Any


class ExecutionMode(str, Enum):
    REAL = "REAL"
    SIMULATION = "SIMULATION"
    DRY_RUN = "DRY_RUN"


class LifecycleStatus(str, Enum):
    PROPOSED = "PROPOSED"
    GOVERNED = "GOVERNED"
    DESIGNED = "DESIGNED"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    IMPLEMENTING = "IMPLEMENTING"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    DECISION = "DECISION"
    MEMORY = "MEMORY"
    
    # Exception / Terminal statuses
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    DISCARDED = "DISCARDED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class IllegalStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class ExecutionModeViolationError(Exception):
    """Raised when an attempt is made to illegally mutate execution mode or treat simulation as real evidence."""
    pass


class ExperimentStateMachine:
    """
    State machine validator enforcing the governed research lifecycle.
    """
    VALID_TRANSITIONS: Dict[LifecycleStatus, Set[LifecycleStatus]] = {
        LifecycleStatus.PROPOSED: {
            LifecycleStatus.GOVERNED,
            LifecycleStatus.BLOCKED,
            LifecycleStatus.DISCARDED
        },
        LifecycleStatus.GOVERNED: {
            LifecycleStatus.DESIGNED,
            LifecycleStatus.BLOCKED,
            LifecycleStatus.DISCARDED
        },
        LifecycleStatus.DESIGNED: {
            LifecycleStatus.REVIEWED,
            LifecycleStatus.BLOCKED,
            LifecycleStatus.DISCARDED
        },
        LifecycleStatus.REVIEWED: {
            LifecycleStatus.APPROVED,
            LifecycleStatus.DISCARDED,
            LifecycleStatus.NEEDS_REVIEW,
            LifecycleStatus.BLOCKED
        },
        LifecycleStatus.APPROVED: {
            LifecycleStatus.IMPLEMENTING,
            LifecycleStatus.BLOCKED
        },
        LifecycleStatus.IMPLEMENTING: {
            LifecycleStatus.RUNNING,
            LifecycleStatus.FAILED,
            LifecycleStatus.BLOCKED
        },
        LifecycleStatus.RUNNING: {
            LifecycleStatus.EVALUATING,
            LifecycleStatus.FAILED
        },
        LifecycleStatus.EVALUATING: {
            LifecycleStatus.COMPLETED,
            LifecycleStatus.FAILED
        },
        LifecycleStatus.COMPLETED: {
            LifecycleStatus.DECISION,
            LifecycleStatus.MEMORY
        },
        LifecycleStatus.DECISION: {
            LifecycleStatus.MEMORY,
            LifecycleStatus.DISCARDED,
            LifecycleStatus.NEEDS_REVIEW
        },
        LifecycleStatus.MEMORY: set(),  # Terminal state
        LifecycleStatus.FAILED: set(),  # Terminal state
        LifecycleStatus.BLOCKED: set(), # Terminal state
        LifecycleStatus.DISCARDED: set(), # Terminal state
        LifecycleStatus.NEEDS_REVIEW: set() # Terminal state
    }

    @classmethod
    def validate_transition(
        cls,
        current_status: LifecycleStatus,
        next_status: LifecycleStatus,
        experiment_id: str = "UNKNOWN"
    ) -> None:
        """
        Validates that transition from current_status to next_status is legally allowed.
        """
        allowed = cls.VALID_TRANSITIONS.get(current_status, set())
        if next_status not in allowed:
            raise IllegalStateTransitionError(
                f"ILLEGAL STATE TRANSITION for {experiment_id}: Cannot transition from "
                f"'{current_status.value}' to '{next_status.value}'. "
                f"Allowed next states from '{current_status.value}': {[s.value for s in allowed]}."
            )

    @classmethod
    def validate_execution_mode_invariance(
        cls,
        existing_mode: str,
        attempted_mode: str,
        experiment_id: str = "UNKNOWN"
    ) -> None:
        """
        Guarantees that a SIMULATION or DRY_RUN record can never be converted into a REAL record.
        """
        ex_mode = existing_mode.upper()
        att_mode = attempted_mode.upper()

        if ex_mode in ("SIMULATION", "DRY_RUN") and att_mode == "REAL":
            raise ExecutionModeViolationError(
                f"CRITICAL INTEGRITY VIOLATION for {experiment_id}: Cannot mutate execution_mode "
                f"from '{existing_mode}' to 'REAL'. Simulated results can never be converted "
                f"into real empirical scientific evidence. A real experiment requires independent "
                f"execution under an approved protocol with a distinct experiment ID."
            )

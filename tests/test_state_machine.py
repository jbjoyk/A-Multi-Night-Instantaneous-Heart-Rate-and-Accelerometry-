"""
tests/test_state_machine.py: Tests for ExperimentStateMachine sequential validation and illegal transition blocks.
"""

import pytest
from research_engine.state_machine import (
    LifecycleStatus, ExecutionMode, ExperimentStateMachine,
    IllegalStateTransitionError, ExecutionModeViolationError
)


def test_valid_sequential_lifecycle_transitions():
    sm = ExperimentStateMachine()
    
    # Valid forward path:
    # PROPOSED -> GOVERNED -> DESIGNED -> REVIEWED -> APPROVED -> IMPLEMENTING -> RUNNING -> EVALUATING -> COMPLETED -> DECISION -> MEMORY
    path = [
        LifecycleStatus.PROPOSED,
        LifecycleStatus.GOVERNED,
        LifecycleStatus.DESIGNED,
        LifecycleStatus.REVIEWED,
        LifecycleStatus.APPROVED,
        LifecycleStatus.IMPLEMENTING,
        LifecycleStatus.RUNNING,
        LifecycleStatus.EVALUATING,
        LifecycleStatus.COMPLETED,
        LifecycleStatus.DECISION,
        LifecycleStatus.MEMORY
    ]

    for i in range(len(path) - 1):
        curr_state = path[i]
        next_state = path[i + 1]
        # Should not raise exception
        sm.validate_transition(curr_state, next_state, "EXP-TEST")


def test_valid_rejection_paths():
    sm = ExperimentStateMachine()

    # Can transition to BLOCKED from GOVERNED
    sm.validate_transition(LifecycleStatus.GOVERNED, LifecycleStatus.BLOCKED, "EXP-TEST")

    # Can transition to DISCARDED from REVIEWED
    sm.validate_transition(LifecycleStatus.REVIEWED, LifecycleStatus.DISCARDED, "EXP-TEST")

    # Can transition to FAILED from RUNNING
    sm.validate_transition(LifecycleStatus.RUNNING, LifecycleStatus.FAILED, "EXP-TEST")


def test_illegal_state_jumps_rejected():
    sm = ExperimentStateMachine()

    # PROPOSED -> COMPLETED (bypassing review and execution)
    with pytest.raises(IllegalStateTransitionError) as exc_info1:
        sm.validate_transition(LifecycleStatus.PROPOSED, LifecycleStatus.COMPLETED, "EXP-TEST")
    assert "ILLEGAL STATE TRANSITION" in str(exc_info1.value)

    # PROPOSED -> RUNNING (bypassing governance & design)
    with pytest.raises(IllegalStateTransitionError) as exc_info2:
        sm.validate_transition(LifecycleStatus.PROPOSED, LifecycleStatus.RUNNING, "EXP-TEST")
    assert "ILLEGAL STATE TRANSITION" in str(exc_info2.value)

    # RUNNING -> MEMORY (bypassing evaluation)
    with pytest.raises(IllegalStateTransitionError) as exc_info3:
        sm.validate_transition(LifecycleStatus.RUNNING, LifecycleStatus.MEMORY, "EXP-TEST")
    assert "ILLEGAL STATE TRANSITION" in str(exc_info3.value)

    # IMPLEMENTING -> COMPLETED
    with pytest.raises(IllegalStateTransitionError) as exc_info4:
        sm.validate_transition(LifecycleStatus.IMPLEMENTING, LifecycleStatus.COMPLETED, "EXP-TEST")
    assert "ILLEGAL STATE TRANSITION" in str(exc_info4.value)


def test_terminal_states_cannot_transition():
    sm = ExperimentStateMachine()

    terminal_states = [
        LifecycleStatus.MEMORY,
        LifecycleStatus.FAILED,
        LifecycleStatus.BLOCKED,
        LifecycleStatus.DISCARDED
    ]

    for term in terminal_states:
        with pytest.raises(IllegalStateTransitionError) as exc_info:
            sm.validate_transition(term, LifecycleStatus.RUNNING, "EXP-TEST")
        assert "ILLEGAL STATE TRANSITION" in str(exc_info.value)


def test_execution_mode_mutation_rejected():
    sm = ExperimentStateMachine()

    # Attempting to convert SIMULATION to REAL
    with pytest.raises(ExecutionModeViolationError) as exc_info:
        sm.validate_execution_mode_invariance("SIMULATION", "REAL", "EXP-015-SIM")
    assert "CRITICAL INTEGRITY VIOLATION" in str(exc_info.value)

    # Attempting to convert DRY_RUN to REAL
    with pytest.raises(ExecutionModeViolationError) as exc_info2:
        sm.validate_execution_mode_invariance("DRY_RUN", "REAL", "EXP-016-DRY")
    assert "CRITICAL INTEGRITY VIOLATION" in str(exc_info2.value)

    # REAL to REAL is allowed
    sm.validate_execution_mode_invariance("REAL", "REAL", "EXP-014")

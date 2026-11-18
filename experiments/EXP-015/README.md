# EXP-015: Circadian/Clock Feature Pruning (Simulation / Protocol Spec)

## Overview
Protocol specification and simulation benchmark evaluating whether pruning cyclic time-of-day features ($\sin/\cos$ clock coordinates) forces models to rely on true autonomic physiology rather than temporal shortcut learning.

## Interface
- Execution Mode: `ExecutionMode.SIMULATION` (Non-evidence protocol specification).
- Governed under `DEC-021` protocol framework.

## Invariants
- `ResearchMemoryValidator`: Marked explicitly as `SIMULATION`; strictly excluded from empirical leaderboards and publication metrics.
- Execution mode is permanently immutable.

## Side Effects
- Records simulated governance token in memory registry.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_simulation_safety.py`

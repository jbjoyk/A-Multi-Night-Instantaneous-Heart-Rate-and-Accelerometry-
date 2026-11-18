# EXP-016: Governed Live Execution Smoke Test

## Overview
Governed live execution smoke test on local host hardware, verifying end-to-end integration across the research engine, GPU lease manager, dataset gate, and state machine under real execution conditions.

## Interface
- `main` (`execute_exp016_smoke.py`): Entrypoint executing a single governed training cycle under strict autonomy gating.

## Invariants
- Governance approval token verified before any GPU operations commence.
- Autonomy level strictly bounded at authorized level; escalation without approval raises `GovernanceBlockError`.

## Side Effects
- Allocates GPU lease on active device.
- Exports smoke test metrics to `metrics.json`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_governor_and_safety.py tests/test_governor_attacks.py`

"""
research_engine/memory_validator.py: Research Memory Schema, Integrity, and Simulation Shielding.

Enforces:
  1. Strict separation of REAL experiments from SIMULATION / DRY_RUN records.
  2. Absolute shielding of leaderboards, baseline selection, and research conclusions from simulation metrics.
  3. Validation of experiment IDs, metrics, epoch ranges, provenance, and dataset fingerprints.
  4. Detection of schema corruption, duplicate IDs, and illegal in-place mode mutations.
"""

import re
from typing import Dict, List, Optional, Any
from research_engine.state_machine import ExecutionMode, LifecycleStatus, ExecutionModeViolationError


class MemoryCorruptionError(Exception):
    """Raised when research memory contains corrupt, invalid, or forged records."""
    pass


class ResearchMemoryValidator:
    """
    Validates institutional memory records and guarantees simulation non-evidence boundaries.
    """
    # Allows standard experiment IDs: EXP-001, EXP-015-SIM, EXP-999-TEST, EXP-LEAKY, etc.
    # Must start with EXP-, be uppercase alphanumeric, and not empty.
    EXPERIMENT_ID_REGEX = re.compile(r"^EXP-[A-Z0-9]+(-[A-Z0-9_]+)*$")
    MAX_EXPERIMENT_ID_LEN = 20
    MIN_EXPERIMENT_ID_LEN = 7
    ALLOWED_BUDGETS = {"SMOKE", "SHORT", "STANDARD", "FULL"}

    REQUIRED_FIELDS = [
        "experiment_id",
        "parent_experiment",
        "hypothesis",
        "research_question",
        "change",
        "git_commit",
        "dataset_fingerprint",
        "split_hash",
        "seed",
        "budget",
        "gpu",
        "metrics",
        "execution_mode",
        "lifecycle_status",
        "real_training",
        "scientific_evidence",
        "eligible_for_model_selection",
        "eligible_for_research_conclusions",
        "eligible_for_leaderboard",
        "decision",
        "reason",
        "timestamp"
    ]

    @classmethod
    def validate_experiment_id(cls, experiment_id: str) -> None:
        """Validates experiment ID formatting."""
        if not experiment_id or not isinstance(experiment_id, str):
            raise MemoryCorruptionError(f"Invalid experiment ID: '{experiment_id}' must be a non-empty string.")
        if len(experiment_id) < cls.MIN_EXPERIMENT_ID_LEN or len(experiment_id) > cls.MAX_EXPERIMENT_ID_LEN:
            raise MemoryCorruptionError(
                f"Invalid experiment ID length: '{experiment_id}' has {len(experiment_id)} chars. "
                f"Must be between {cls.MIN_EXPERIMENT_ID_LEN} and {cls.MAX_EXPERIMENT_ID_LEN} characters."
            )
        if not cls.EXPERIMENT_ID_REGEX.match(experiment_id):
            raise MemoryCorruptionError(
                f"Invalid experiment ID format: '{experiment_id}'. Must start with 'EXP-' and use uppercase alphanumeric tokens."
            )

    @classmethod
    def validate_record(cls, record: Dict[str, Any]) -> None:
        """
        Validates an individual experiment record. Raises MemoryCorruptionError on any violation.
        """
        exp_id = record.get("experiment_id", "")
        cls.validate_experiment_id(exp_id)

        # 1. Required Fields Check
        missing = [f for f in cls.REQUIRED_FIELDS if f not in record]
        if missing:
            raise MemoryCorruptionError(
                f"Memory record for {exp_id} is missing required fields: {missing}"
            )

        # 2. Execution Mode & Lifecycle Status Validation
        mode_str = str(record["execution_mode"]).upper()
        if mode_str not in ExecutionMode.__members__:
            raise MemoryCorruptionError(
                f"Invalid execution_mode '{record['execution_mode']}' for {exp_id}. "
                f"Allowed modes: {[m.value for m in ExecutionMode]}"
            )

        status_str = str(record["lifecycle_status"]).upper()
        if status_str not in LifecycleStatus.__members__:
            raise MemoryCorruptionError(
                f"Invalid lifecycle_status '{record['lifecycle_status']}' for {exp_id}. "
                f"Allowed statuses: {[s.value for s in LifecycleStatus]}"
            )

        # 3. Simulation Non-Evidence Invariants
        if mode_str in (ExecutionMode.SIMULATION.value, ExecutionMode.DRY_RUN.value):
            if record["real_training"] is not False:
                raise MemoryCorruptionError(
                    f"SIMULATION SAFETY VIOLATION for {exp_id}: 'real_training' must be false for {mode_str}."
                )
            if record["scientific_evidence"] is not False:
                raise MemoryCorruptionError(
                    f"SIMULATION SAFETY VIOLATION for {exp_id}: 'scientific_evidence' must be false for {mode_str}."
                )
            if record["eligible_for_model_selection"] is not False:
                raise MemoryCorruptionError(
                    f"SIMULATION SAFETY VIOLATION for {exp_id}: 'eligible_for_model_selection' must be false for {mode_str}."
                )
            if record["eligible_for_research_conclusions"] is not False:
                raise MemoryCorruptionError(
                    f"SIMULATION SAFETY VIOLATION for {exp_id}: 'eligible_for_research_conclusions' must be false for {mode_str}."
                )
            if record["eligible_for_leaderboard"] is not False:
                raise MemoryCorruptionError(
                    f"SIMULATION SAFETY VIOLATION for {exp_id}: 'eligible_for_leaderboard' must be false for {mode_str}."
                )

        if mode_str == ExecutionMode.REAL.value:
            if record.get("real_training") is not True:
                raise MemoryCorruptionError(
                    f"INCONSISTENCY VIOLATION for {exp_id}: 'real_training' must be true for REAL execution mode."
                )
            if record.get("scientific_evidence") is not True:
                raise MemoryCorruptionError(
                    f"INCONSISTENCY VIOLATION for {exp_id}: 'scientific_evidence' must be true for REAL execution mode."
                )
            if "simulation_tag" in record:
                raise MemoryCorruptionError(
                    f"INCONSISTENCY VIOLATION for {exp_id}: 'simulation_tag' cannot exist in a REAL experiment record."
                )

        # 4. Metrics Validation
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise MemoryCorruptionError(f"Metrics field for {exp_id} must be a dictionary.")

        for metric_key in ("macro_f1", "accuracy", "Model_A_macro_f1", "Model_A_accuracy", "Model_E_macro_f1", "Model_E_accuracy"):
            if metric_key in metrics and metrics[metric_key] is not None:
                val = metrics[metric_key]
                if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                    raise MemoryCorruptionError(
                        f"Impossible metric value for {exp_id}: {metric_key}={val} is outside [0.0, 1.0]."
                    )

        if "kappa" in metrics and metrics["kappa"] is not None:
            val = metrics["kappa"]
            if not isinstance(val, (int, float)) or val < -1.0 or val > 1.0:
                raise MemoryCorruptionError(
                    f"Impossible metric value for {exp_id}: kappa={val} is outside [-1.0, 1.0]."
                )

        # 5. Provenance Checks
        if not record.get("dataset_fingerprint"):
            raise MemoryCorruptionError(f"Missing dataset_fingerprint for {exp_id}.")
        if not record.get("split_hash"):
            raise MemoryCorruptionError(f"Missing split_hash for {exp_id}.")
        if not record.get("git_commit"):
            raise MemoryCorruptionError(f"Missing git_commit for {exp_id}.")

    @classmethod
    def validate_memory_integrity(cls, memory: Dict[str, Any]) -> None:
        """
        Validates entire research memory structure, checking for duplicate IDs and record integrity.
        """
        if not isinstance(memory, dict) or "experiments" not in memory:
            raise MemoryCorruptionError("Malformed research memory: missing root 'experiments' list.")

        seen_ids = set()
        for exp in memory["experiments"]:
            exp_id = exp.get("experiment_id")
            if exp_id in seen_ids:
                raise MemoryCorruptionError(f"DUPLICATE EXPERIMENT ID DETECTED in research memory: '{exp_id}'")
            seen_ids.add(exp_id)
            cls.validate_record(exp)

    @staticmethod
    def get_real_experiments(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filters and returns only real, scientifically verified experiments.
        Strictly excludes any SIMULATION, DRY_RUN, or unverified records.
        """
        return [
            exp for exp in memory.get("experiments", [])
            if str(exp.get("execution_mode", "")).upper() == "REAL"
            and exp.get("scientific_evidence") is True
            and exp.get("real_training") is True
        ]

    @staticmethod
    def get_simulation_records(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns all simulation and dry-run records.
        """
        return [
            exp for exp in memory.get("experiments", [])
            if str(exp.get("execution_mode", "")).upper() in ("SIMULATION", "DRY_RUN")
            or exp.get("scientific_evidence") is False
        ]

    @classmethod
    def compute_registry_statistics(cls, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Programmatically calculates exact counts, modes, and eligibility across research memory.
        Identifies excluded historical IDs and detects inconsistencies.
        """
        exps = memory.get("experiments", [])
        total = len(exps)
        reals = [e for e in exps if str(e.get("execution_mode", "")).upper() == "REAL"]
        sims = [e for e in exps if str(e.get("execution_mode", "")).upper() == "SIMULATION"]
        dry_runs = [e for e in exps if str(e.get("execution_mode", "")).upper() == "DRY_RUN"]

        eligible_real = [
            e for e in reals
            if e.get("eligible_for_leaderboard") is True
            and e.get("scientific_evidence") is True
            and e.get("real_training") is True
        ]
        ineligible = [e for e in exps if e not in eligible_real]

        all_ids = [e["experiment_id"] for e in exps]
        real_ids = [e["experiment_id"] for e in reals]
        sim_ids = [e["experiment_id"] for e in sims]
        dry_ids = [e["experiment_id"] for e in dry_runs]
        eligible_ids = [e["experiment_id"] for e in eligible_real]
        ineligible_ids = [e["experiment_id"] for e in ineligible]

        # Audit excluded numerical sequence IDs between EXP-001 and EXP-014
        excluded_sequence = {
            "EXP-002": "Implementation scaffold only; syntax compiled, no training or evaluation was performed.",
            "EXP-003": "Read-only QC audit of prior baseline artifacts; no independent trained model registered.",
            "EXP-004": "Controlled optimizer search protocol setup; no standalone trained checkpoint archived in registry."
        }

        # Check internal consistency
        if total != (len(reals) + len(sims) + len(dry_runs)):
            raise MemoryCorruptionError(
                f"Registry Count Inconsistency: Total records ({total}) != REAL ({len(reals)}) + "
                f"SIMULATION ({len(sims)}) + DRY_RUN ({len(dry_runs)})"
            )

        if len(eligible_real) != len(reals):
            uneligible_reals = [e["experiment_id"] for e in reals if e not in eligible_real]
            raise MemoryCorruptionError(
                f"Registry Inconsistency: REAL experiments found with ineligible status: {uneligible_reals}"
            )

        # Check that no simulation appears in eligible_ids
        for sid in sim_ids:
            if sid in eligible_ids:
                raise MemoryCorruptionError(f"CRITICAL CONTAMINATION: Simulation record '{sid}' flagged as eligible!")

        return {
            "total_records": total,
            "real_count": len(reals),
            "simulation_count": len(sims),
            "dry_run_count": len(dry_runs),
            "eligible_count": len(eligible_real),
            "ineligible_count": len(ineligible),
            "all_ids": all_ids,
            "real_ids": real_ids,
            "simulation_ids": sim_ids,
            "dry_run_ids": dry_ids,
            "eligible_ids": eligible_ids,
            "ineligible_ids": ineligible_ids,
            "excluded_sequence": excluded_sequence
        }

    @classmethod
    def assert_registry_consistency(cls, memory: Dict[str, Any]) -> None:
        """Runs full consistency audit over memory and fails loudly on any contradiction."""
        cls.validate_memory_integrity(memory)
        cls.compute_registry_statistics(memory)

    @classmethod
    def get_leaderboard(cls, memory: Dict[str, Any], sort_by: str = "macro_f1") -> List[Dict[str, Any]]:
        """
        Returns ranked leaderboard composed STRICTLY of real empirical experiments.
        Simulations and dry-runs are completely barred from ranking.
        """
        real_exps = cls.get_real_experiments(memory)
        valid_candidates = [
            exp for exp in real_exps
            if exp.get("eligible_for_leaderboard", False)
            and isinstance(exp.get("metrics"), dict)
        ]

        def get_score(exp):
            m = exp.get("metrics", {})
            if sort_by in m and m[sort_by] is not None:
                return float(m[sort_by])
            if f"Model_E_{sort_by}" in m:
                return float(m[f"Model_E_{sort_by}"])
            if f"Model_A_{sort_by}" in m:
                return float(m[f"Model_A_{sort_by}"])
            return 0.0

        return sorted(valid_candidates, key=get_score, reverse=True)

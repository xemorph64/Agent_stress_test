"""
envs/agent_stress_test_env/server/graders.py
---------------------------------------------
Grading logic for the stress test environment.

Based on MAST research (NeurIPS 2025):
- Specification & System Design: 41.8% of failures
- Inter-Agent Misalignment: 36.9% of failures
- Task Verification: 21.3% of failures

Provides deterministic graders that score 0.0-1.0 based on:
- Diagnosis quality (partial credit for correct identification)
- Resilience mechanism application
- Stress test pass/fail (10 trials)
"""

from __future__ import annotations

from typing import Any

from .workflow_simulator import (
    ResilienceConfig,
    WorkflowSimulator,
    create_easy_task,
    create_hard_task,
    create_medium_task,
    create_termination_task,
    create_memory_task,
    create_reasoning_task,
)


class Grader:
    """Base grader class for stress tests."""

    def __init__(self, task_id: str, difficulty: str):
        self.task_id = task_id
        self.difficulty = difficulty

    def _parse_diagnosis(self, diagnosis: str) -> dict[str, float]:
        """
        Parse agent's diagnosis for partial credit.

        Returns dict with keyword scores for partial credit.
        """
        diagnosis_lower = diagnosis.lower()
        scores = {
            "spec_issue": 0.0,
            "format_issue": 0.0,
            "verification_issue": 0.0,
            "misinterpretation": 0.0,
            "mismatch": 0.0,
            "premature": 0.0,
            "contradiction": 0.0,
            "vague": 0.0,
            "ambiguous": 0.0,
            "yaml": 0.0,
            "json": 0.0,
            "translate": 0.0,
            "verify": 0.0,
            "check": 0.0,
            "review": 0.0,
            "termination": 0.0,
            "loop": 0.0,
            "memory": 0.0,
            "context": 0.0,
            "reasoning": 0.0,
            "action": 0.0,
        }

        # Specification keywords
        if (
            "spec" in diagnosis_lower
            or "vague" in diagnosis_lower
            or "ambiguous" in diagnosis_lower
        ):
            scores["spec_issue"] = 0.10
        if "misinterpret" in diagnosis_lower or "unclear" in diagnosis_lower:
            scores["misinterpretation"] = 0.10

        # Format keywords
        if (
            "format" in diagnosis_lower
            or "yaml" in diagnosis_lower
            or "json" in diagnosis_lower
        ):
            scores["format_issue"] = 0.10
        if "mismatch" in diagnosis_lower or "translate" in diagnosis_lower:
            scores["mismatch"] = 0.10

        # Verification keywords
        if "verif" in diagnosis_lower or "check" in diagnosis_lower:
            scores["verification_issue"] = 0.10
        if "premature" in diagnosis_lower or "incomplete" in diagnosis_lower:
            scores["premature"] = 0.10
        if "contradict" in diagnosis_lower:
            scores["contradiction"] = 0.10

        # Termination keywords (IBM 2026 - FATAL)
        if "terminat" in diagnosis_lower or "loop" in diagnosis_lower:
            scores["termination"] = 0.10
        if "infinite" in diagnosis_lower or "repeat" in diagnosis_lower:
            scores["loop"] = 0.10

        # Memory/Context keywords (IBM 2026 - FATAL)
        if "memory" in diagnosis_lower or "forget" in diagnosis_lower:
            scores["memory"] = 0.10
        if "context" in diagnosis_lower or "history" in diagnosis_lower:
            scores["context"] = 0.10

        # Reasoning-Action keywords (IBM 2026 - FATAL)
        if "reason" in diagnosis_lower or "think" in diagnosis_lower:
            scores["reasoning"] = 0.10
        if "action" in diagnosis_lower or "execut" in diagnosis_lower:
            scores["action"] = 0.10

        return scores

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        """
        Grade the agent's response.

        Args:
            agent_config: Configuration the agent applied
            task_description: Description of the task
            failure_mode: Description of the failure mode
            diagnosis: Agent's diagnosis of the problem

        Returns:
            Tuple of (score 0.0-1.0, details dict)
        """
        raise NotImplementedError


class EasyGrader(Grader):
    """
    Grader for easy task: Specification Ambiguity (MAST Category: Specification & System Design)

    Task: Agent receives vague role definition causing task misinterpretation.
    Fix: Must provide explicit role specification (spec_fix field).
    """

    def __init__(self):
        super().__init__("easy", "easy")
        self._required_keywords = [
            "spec",
            "role",
            "explicit",
            "capabilities",
            "constraints",
            "success_criteria",
        ]

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        nodes, _, _ = create_easy_task()

        resilience = self._parse_config(agent_config, diagnosis)
        simulator = WorkflowSimulator(nodes, seed=42)

        results = []
        for _ in range(10):
            result = simulator.run_workflow(resilience)
            results.append(result.success)

        success_rate = sum(results) / len(results)

        # Check for explicit spec fix
        has_spec_fix = bool(agent_config.get("spec_fix") or "")
        has_explicit_spec = agent_config.get("explicit_role_spec", False)

        # Check diagnosis for partial credit
        diagnosis_scores = self._parse_diagnosis(diagnosis)
        diagnosis_points = min(
            0.25, diagnosis_scores["spec_issue"] + diagnosis_scores["misinterpretation"]
        )

        score = 0.0

        # Points for proper specification
        if has_spec_fix or has_explicit_spec:
            score += 0.35

        # Points for simulation success
        if success_rate > 0:
            score += success_rate * 0.40

        # Bonus for high success rate
        if success_rate >= 0.8:
            score += 0.10

        # Partial credit from diagnosis
        score += diagnosis_points

        score = min(1.0, max(0.0, score))

        return score, {
            "success_rate": success_rate,
            "has_spec_fix": has_spec_fix,
            "has_explicit_spec": has_explicit_spec,
            "diagnosis_points": diagnosis_points,
            "config": agent_config,
            "diagnosis": diagnosis,
        }

    def _parse_config(
        self, agent_config: dict[str, Any], diagnosis: str
    ) -> ResilienceConfig:
        return ResilienceConfig(
            retry_max=agent_config.get("retry_max", 0),
            retry_delay_ms=agent_config.get("retry_delay_ms", 0),
            timeout_ms=agent_config.get("timeout_ms", 30000),
            fallback=agent_config.get("fallback", "abort"),
            circuit_breaker_threshold=agent_config.get(
                "circuit_breaker_threshold", 1.0
            ),
            context_strategy=agent_config.get("context_strategy", "truncate"),
            context_summarization_threshold=agent_config.get(
                "context_summarization_threshold", 500
            ),
            min_review_depth=agent_config.get("min_review_depth", 1),
            consistency_check=agent_config.get("consistency_check", False),
            diagnosis=agent_config.get("diagnosis", ""),
            spec_fix=agent_config.get("spec_fix", ""),
            format_translator=agent_config.get("format_translator", False),
        )


class MediumGrader(Grader):
    """
    Grader for medium task: Format Mismatch (MAST Category: Inter-Agent Misalignment)

    Task: Planner outputs YAML but Executor expects JSON - format mismatch.
    Fix: Must add format translation (format_translator field).
    """

    def __init__(self):
        super().__init__("medium", "medium")

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        nodes, _, _ = create_medium_task()

        resilience = self._parse_config(agent_config, diagnosis)
        simulator = WorkflowSimulator(nodes, seed=42)

        results = []
        latencies = []

        for _ in range(10):
            result = simulator.run_workflow(resilience)
            results.append(result.success)
            latencies.append(result.total_latency_ms)

        success_rate = sum(results) / len(results)

        # Check for format translator
        has_format_translator = agent_config.get("format_translator", False)

        # Check diagnosis for partial credit
        diagnosis_scores = self._parse_diagnosis(diagnosis)
        diagnosis_points = min(
            0.20, diagnosis_scores["format_issue"] + diagnosis_scores["mismatch"]
        )

        score = 0.0

        # Points for format translator
        if has_format_translator:
            score += 0.30

        # Points for simulation success
        if success_rate > 0:
            score += success_rate * 0.45

        # Bonus for high success rate
        if success_rate >= 0.7:
            score += 0.10

        # Partial credit from diagnosis
        score += diagnosis_points

        score = min(1.0, max(0.0, score))

        return score, {
            "success_rate": success_rate,
            "has_format_translator": has_format_translator,
            "diagnosis_points": diagnosis_points,
            "avg_latency": sum(latencies) / len(latencies),
            "config": agent_config,
            "diagnosis": diagnosis,
        }

    def _parse_config(
        self, agent_config: dict[str, Any], diagnosis: str
    ) -> ResilienceConfig:
        return ResilienceConfig(
            retry_max=agent_config.get("retry_max", 0),
            retry_delay_ms=agent_config.get("retry_delay_ms", 0),
            timeout_ms=agent_config.get("timeout_ms", 30000),
            fallback=agent_config.get("fallback", "abort"),
            circuit_breaker_threshold=agent_config.get(
                "circuit_breaker_threshold", 1.0
            ),
            context_strategy=agent_config.get("context_strategy", "truncate"),
            context_summarization_threshold=agent_config.get(
                "context_summarization_threshold", 500
            ),
            min_review_depth=agent_config.get("min_review_depth", 1),
            consistency_check=agent_config.get("consistency_check", False),
            diagnosis=agent_config.get("diagnosis", ""),
            spec_fix=agent_config.get("spec_fix", ""),
            format_translator=agent_config.get("format_translator", False),
        )


class HardGrader(Grader):
    """
    Grader for hard task: Verification Failure (MAST Category: Task Verification)

    Task: Writer produces contradictions, reviewer prematurely approves without checks.
    Fix: Must add multi-level verification (consistency_check, min_review_depth).
    """

    def __init__(self):
        super().__init__("hard", "hard")

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        nodes, _, _ = create_hard_task()

        resilience = self._parse_config(agent_config, diagnosis)
        simulator = WorkflowSimulator(nodes, seed=42)

        results = []

        for _ in range(10):
            result = simulator.run_workflow(resilience)
            results.append(result.success)

        success_rate = sum(results) / len(results)

        # Check for verification mechanisms
        has_consistency_check = agent_config.get("consistency_check", False)
        has_review_depth = agent_config.get("min_review_depth", 1) >= 3

        # Check diagnosis for partial credit
        diagnosis_scores = self._parse_diagnosis(diagnosis)
        diagnosis_points = min(
            0.20,
            diagnosis_scores["verification_issue"]
            + diagnosis_scores["premature"]
            + diagnosis_scores["contradiction"],
        )

        score = 0.0

        # Points for verification mechanisms
        if has_consistency_check:
            score += 0.15
        if has_review_depth:
            score += 0.15

        # Points for simulation success
        if success_rate > 0:
            score += success_rate * 0.45

        # Bonus for reasonable success rate
        if success_rate >= 0.5:
            score += 0.10

        # Partial credit from diagnosis
        score += diagnosis_points

        score = min(1.0, max(0.0, score))

        return score, {
            "success_rate": success_rate,
            "has_consistency_check": has_consistency_check,
            "has_review_depth": has_review_depth,
            "diagnosis_points": diagnosis_points,
            "config": agent_config,
            "diagnosis": diagnosis,
        }

    def _parse_config(
        self, agent_config: dict[str, Any], diagnosis: str
    ) -> ResilienceConfig:
        return ResilienceConfig(
            retry_max=agent_config.get("retry_max", 0),
            retry_delay_ms=agent_config.get("retry_delay_ms", 0),
            timeout_ms=agent_config.get("timeout_ms", 30000),
            fallback=agent_config.get("fallback", "abort"),
            circuit_breaker_threshold=agent_config.get(
                "circuit_breaker_threshold", 1.0
            ),
            context_strategy=agent_config.get("context_strategy", "truncate"),
            context_summarization_threshold=agent_config.get(
                "context_summarization_threshold", 500
            ),
            min_review_depth=agent_config.get("min_review_depth", 5),
            consistency_check=agent_config.get("consistency_check", True),
            diagnosis=agent_config.get("diagnosis", ""),
            spec_fix=agent_config.get("spec_fix", ""),
            format_translator=agent_config.get("format_translator", False),
        )


def get_grader(task_id: str) -> Grader:
    """Get the appropriate grader for a task."""
    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
        "termination": TerminationGrader(),
        "memory": MemoryGrader(),
        "reasoning": ReasoningGrader(),
    }
    return graders.get(task_id, EasyGrader())


class TerminationGrader(Grader):
    """
    Grader for termination task: FM-1.5/FM-3.1 (IBM 2026 - FATAL FAILURE)

    Task: Agent struggles to recognize task completion - loops or prematurely exits.
    Fix: Implement explicit termination conditions with success criteria.
    """

    def __init__(self):
        super().__init__("termination", "medium")

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        from .workflow_simulator import create_termination_task

        nodes, _, _ = create_termination_task()
        resilience = self._parse_config(agent_config, diagnosis)
        simulator = WorkflowSimulator(nodes, seed=42)

        results = []
        for _ in range(10):
            result = simulator.run_workflow(resilience)
            results.append(result.success)

        success_rate = sum(results) / len(results)

        has_termination_detection = agent_config.get("explicit_termination", False)
        has_max_iterations = agent_config.get("max_iterations", 0) > 0

        diagnosis_scores = self._parse_diagnosis(diagnosis)
        diagnosis_points = min(
            0.15, diagnosis_scores["termination"] + diagnosis_scores["loop"]
        )

        score = 0.0

        if has_termination_detection:
            score += 0.25
        if has_max_iterations:
            score += 0.20

        if success_rate > 0:
            score += success_rate * 0.30

        if success_rate >= 0.6:
            score += 0.15

        score += diagnosis_points

        score = min(1.0, max(0.0, score))

        return score, {
            "success_rate": success_rate,
            "has_termination_detection": has_termination_detection,
            "has_max_iterations": has_max_iterations,
            "diagnosis_points": diagnosis_points,
            "config": agent_config,
            "diagnosis": diagnosis,
        }

    def _parse_config(
        self, agent_config: dict[str, Any], diagnosis: str
    ) -> ResilienceConfig:
        return ResilienceConfig(
            retry_max=agent_config.get("max_iterations", 50),
            retry_delay_ms=agent_config.get("retry_delay_ms", 0),
            timeout_ms=agent_config.get("timeout_ms", 30000),
            fallback=agent_config.get("fallback", "abort"),
            circuit_breaker_threshold=agent_config.get(
                "circuit_breaker_threshold", 1.0
            ),
            context_strategy=agent_config.get("context_strategy", "truncate"),
            context_summarization_threshold=agent_config.get(
                "context_summarization_threshold", 500
            ),
            min_review_depth=agent_config.get("min_review_depth", 1),
            consistency_check=agent_config.get("consistency_check", False),
            explicit_termination=agent_config.get("explicit_termination", False),
            diagnosis=agent_config.get("diagnosis", ""),
        )


class MemoryGrader(Grader):
    """
    Grader for memory task: FM-1.4 (IBM 2026 - FATAL FAILURE)

    Task: Agent loses conversation history in long traces - forgets original task.
    Fix: Implement context management (sliding window, summarization, state machine).
    """

    def __init__(self):
        super().__init__("memory", "hard")

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        from .workflow_simulator import create_memory_task

        nodes, _, _ = create_memory_task()
        resilience = self._parse_config(agent_config, diagnosis)
        simulator = WorkflowSimulator(nodes, seed=42)

        results = []
        for _ in range(10):
            result = simulator.run_workflow(resilience)
            results.append(result.success)

        success_rate = sum(results) / len(results)

        has_summarization = agent_config.get("context_summarization", False)
        has_sliding_window = agent_config.get("sliding_window", False)

        diagnosis_scores = self._parse_diagnosis(diagnosis)
        diagnosis_points = min(
            0.15, diagnosis_scores["memory"] + diagnosis_scores["context"]
        )

        score = 0.0

        if has_summarization:
            score += 0.20
        if has_sliding_window:
            score += 0.20

        if success_rate > 0:
            score += success_rate * 0.35

        if success_rate >= 0.5:
            score += 0.15

        score += diagnosis_points

        score = min(1.0, max(0.0, score))

        return score, {
            "success_rate": success_rate,
            "has_summarization": has_summarization,
            "has_sliding_window": has_sliding_window,
            "diagnosis_points": diagnosis_points,
            "config": agent_config,
            "diagnosis": diagnosis,
        }

    def _parse_config(
        self, agent_config: dict[str, Any], diagnosis: str
    ) -> ResilienceConfig:
        return ResilienceConfig(
            retry_max=agent_config.get("retry_max", 0),
            retry_delay_ms=agent_config.get("retry_delay_ms", 0),
            timeout_ms=agent_config.get("timeout_ms", 30000),
            fallback=agent_config.get("fallback", "abort"),
            circuit_breaker_threshold=agent_config.get(
                "circuit_breaker_threshold", 1.0
            ),
            context_strategy=agent_config.get("context_strategy", "summarize"),
            context_summarization_threshold=agent_config.get(
                "context_summarization_threshold", 200
            ),
            min_review_depth=agent_config.get("min_review_depth", 1),
            consistency_check=agent_config.get("consistency_check", False),
            context_summarization=agent_config.get("context_summarization", False),
            sliding_window=agent_config.get("sliding_window", False),
            diagnosis=agent_config.get("diagnosis", ""),
        )


class ReasoningGrader(Grader):
    """
    Grader for reasoning-action alignment: FM-2.6 (IBM 2026 - FATAL FAILURE)

    Task: Agent describes correct plan but executes unrelated/redundant command.
    Fix: Implement action validation layer checking execution against reasoning.
    """

    def __init__(self):
        super().__init__("reasoning", "hard")

    def grade(
        self,
        agent_config: dict[str, Any],
        task_description: str,
        failure_mode: str,
        diagnosis: str,
    ) -> tuple[float, dict[str, Any]]:
        from .workflow_simulator import create_reasoning_task

        nodes, _, _ = create_reasoning_task()
        resilience = self._parse_config(agent_config, diagnosis)
        simulator = WorkflowSimulator(nodes, seed=42)

        results = []
        for _ in range(10):
            result = simulator.run_workflow(resilience)
            results.append(result.success)

        success_rate = sum(results) / len(results)

        has_action_validation = agent_config.get("action_validation", False)
        has_consistency_check = agent_config.get("reasoning_consistency_check", False)

        diagnosis_scores = self._parse_diagnosis(diagnosis)
        diagnosis_points = min(
            0.15, diagnosis_scores["reasoning"] + diagnosis_scores["action"]
        )

        score = 0.0

        if has_action_validation:
            score += 0.20
        if has_consistency_check:
            score += 0.20

        if success_rate > 0:
            score += success_rate * 0.35

        if success_rate >= 0.45:
            score += 0.15

        score += diagnosis_points

        score = min(1.0, max(0.0, score))

        return score, {
            "success_rate": success_rate,
            "has_action_validation": has_action_validation,
            "has_consistency_check": has_consistency_check,
            "diagnosis_points": diagnosis_points,
            "config": agent_config,
            "diagnosis": diagnosis,
        }

    def _parse_config(
        self, agent_config: dict[str, Any], diagnosis: str
    ) -> ResilienceConfig:
        return ResilienceConfig(
            retry_max=agent_config.get("retry_max", 0),
            retry_delay_ms=agent_config.get("retry_delay_ms", 0),
            timeout_ms=agent_config.get("timeout_ms", 30000),
            fallback=agent_config.get("fallback", "abort"),
            circuit_breaker_threshold=agent_config.get(
                "circuit_breaker_threshold", 1.0
            ),
            context_strategy=agent_config.get("context_strategy", "truncate"),
            context_summarization_threshold=agent_config.get(
                "context_summarization_threshold", 500
            ),
            min_review_depth=agent_config.get("min_review_depth", 1),
            consistency_check=agent_config.get("consistency_check", False),
            action_validation=agent_config.get("action_validation", False),
            reasoning_consistency_check=agent_config.get(
                "reasoning_consistency_check", False
            ),
            diagnosis=agent_config.get("diagnosis", ""),
        )

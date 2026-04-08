"""
envs/agent_stress_test_env/models.py
--------------------------------------
Action/Observation/State types for the Agentic System Stress Tester environment.

Based on MAST research (NeurIPS 2025) - Multi-agent LLM systems fail 41-86.7% of the time.
Failure categories:
- Specification & System Design: 41.8%
- Inter-Agent Misalignment: 36.9%
- Task Verification: 21.3%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from openenv.core.env_server.interfaces import Action, Observation, State


class ResilienceConfig(Action):
    """
    Agent's configuration for resilience mechanisms.

    The agent outputs this to fix multi-agent workflow failures.
    Supports different fix types based on failure mode:

    MAST Categories (NeurIPS 2025):
    - FC1: System Design (41.8%) - spec, termination, memory
    - FC2: Inter-Agent Misalignment (36.9%) - format, reasoning-action
    - FC3: Task Verification (21.3%) - verification checks

    IBM 2026 Updates:
    - FM-1.5/FM-3.1: Termination awareness (FATAL)
    - FM-1.4: Memory/Context loss (FATAL)
    - FM-2.6: Reasoning-action mismatch (FATAL)
    """

    retry_max: int = 0
    retry_delay_ms: int = 0
    timeout_ms: int = 30000
    fallback: Literal["skip", "summarize", "abort", "retry_last"] = "abort"
    circuit_breaker_threshold: float = 1.0
    context_strategy: Literal["truncate", "summarize", "chunk"] = "truncate"
    context_summarization_threshold: int = 500
    min_review_depth: int = 1
    consistency_check: bool = False

    # MAST FC1: System Design (Easy task - spec ambiguity)
    spec_fix: str = ""
    explicit_role_spec: bool = False

    # MAST FC2: Inter-Agent Misalignment (Medium task - format mismatch)
    format_translator: bool = False

    # MAST FC3: Task Verification (Hard task - verification failure)
    # (uses consistency_check + min_review_depth)

    # IBM 2026: FC1 - Termination Awareness (FATAL)
    explicit_termination: bool = False
    max_iterations: int = 0

    # IBM 2026: FC1 - Memory/Context Management (FATAL)
    context_summarization: bool = False
    sliding_window: bool = False

    # IBM 2026: FC2 - Reasoning-Action Alignment (FATAL)
    action_validation: bool = False
    reasoning_consistency_check: bool = False

    # Agent's diagnosis of the failure mode
    diagnosis: str = ""

    # Metadata (used by server, not passed to model)
    metadata: dict = field(default_factory=dict)


class StressTestObservation(Observation):
    """
    Result of a stress test execution.
    """

    task_id: str = ""
    task_description: str = ""
    scenario_setup: str = ""
    failure_mode_detected: bool = False
    failure_mode_description: str = ""
    failure_category: str = ""  # MAST category: spec, inter_agent, verification
    resilience_applied: bool = False
    applied_config: str = ""
    test_passed: bool = False
    test_completions: int = 0
    test_total_trials: int = 10
    test_latency_ms: int = 0
    diagnosis: str = ""
    diagnosis_points: float = 0.0  # Partial credit from keyword matching
    reward: float = 0.0
    done: bool = False
    error_details: Optional[str] = None


class StressTestState(State):
    """
    State for the stress test environment.
    """

    episode_id: str = ""
    task_id: str = "easy"
    current_task_index: int = 0
    attempts: int = 0
    total_score: float = 0.0
    task_scores: list[float] = []
    step_count: int = 0

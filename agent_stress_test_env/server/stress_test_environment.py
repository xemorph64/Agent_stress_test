"""
envs/agent_stress_test_env/server/stress_test_environment.py
--------------------------------------------------------------
Main environment implementation for Agentic System Stress Tester.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

try:
    from openenv.core.env_server.interfaces import (
        Action,
        Observation,
        State,
        Environment,
    )
    from ..models import ResilienceConfig, StressTestObservation, StressTestState
    from .graders import get_grader
    from .workflow_simulator import (
        create_easy_task,
        create_hard_task,
        create_medium_task,
        create_termination_task,
        create_memory_task,
        create_reasoning_task,
    )
except ImportError:
    from openenv.core.env_server.interfaces import (
        Action,
        Observation,
        State,
        Environment,
    )
    from agent_stress_test_env.models import (
        ResilienceConfig,
        StressTestObservation,
        StressTestState,
    )
    from server.graders import get_grader
    from server.workflow_simulator import (
        create_easy_task,
        create_hard_task,
        create_medium_task,
    )


TASK_DEFINITIONS = {
    "easy": {
        "id": "easy",
        "difficulty": "easy",
        "category": "MAST FC1: System Design (41.8% of failures)",
        "description": "The researcher agent has a vague role definition ('You are a helpful assistant'). This causes task misinterpretation. Your task: Provide an explicit role specification JSON with clear capabilities, constraints, and success criteria.",
        "failure_mode": "FM-1.1: Specification ambiguity - vague role definition causes task misinterpretation",
    },
    "medium": {
        "id": "medium",
        "difficulty": "medium",
        "category": "MAST FC2: Inter-Agent Misalignment (36.9% of failures)",
        "description": "Multi-agent workflow where the planner outputs YAML but the executor expects JSON. This format mismatch causes the executor to fail. Your task: Add a format translation layer/middleware.",
        "failure_mode": "FM-2.x: Format mismatch - planner outputs YAML, executor expects JSON",
    },
    "hard": {
        "id": "hard",
        "difficulty": "hard",
        "category": "MAST FC3: Task Verification (21.3% of failures)",
        "description": "Multi-agent pipeline with verification failure. Writer produces contradictions (30%), reviewer prematurely approves (60%) without checks. Your task: Implement multi-level verification. IBM 2026: FM-3.3 is strongest failure predictor.",
        "failure_mode": "FM-3.1/FM-3.3: Verification failure - premature termination + incorrect verification",
    },
    "termination": {
        "id": "termination",
        "difficulty": "medium",
        "category": "MAST FC1: System Design - FATAL FAILURE",
        "description": "The agent struggles to recognize when a task is complete. It loops indefinitely or prematurely exits. Based on IBM 2026: Kimi-K2 shows +46% spike in termination issues. Your task: Implement explicit termination conditions with success criteria.",
        "failure_mode": "FM-1.5/FM-3.1: Unaware of termination + premature termination",
    },
    "memory": {
        "id": "memory",
        "difficulty": "hard",
        "category": "MAST FC1: System Design - FATAL FAILURE",
        "description": "As conversation history grows, the agent loses context and derails. Based on IBM 2026: GPT-OSS-120B shows 24% memory loss in long traces. Your task: Implement context management - sliding window, summarization, or state machine.",
        "failure_mode": "FM-1.4: Loss of conversation history - agent forgets original task",
    },
    "reasoning": {
        "id": "reasoning",
        "difficulty": "hard",
        "category": "MAST FC2: Inter-Agent Misalignment - FATAL FAILURE",
        "description": "The agent describes correct plan but executes unrelated command. Based on IBM 2026: 92% of Kimi-K2 failures and 94% of GPT-OSS-120B failures show this. Your task: Implement action validation layer checking execution against reasoning.",
        "failure_mode": "FM-2.6: Reasoning-action mismatch - correct thinking, wrong execution",
    },
}


class StressTestEnvironment(
    Environment[ResilienceConfig, StressTestObservation, StressTestState]
):
    """
    Environment for testing multi-agent workflow resilience.

    Tasks progress from easy (single failure) to hard (cascading failures).
    The agent must diagnose the failure mode and apply appropriate resilience.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize the stress test environment."""
        super().__init__()
        self._state = StressTestState()
        self._current_task_index = 0
        self._task_ids = [
            "easy",
            "medium",
            "hard",
            "termination",
            "memory",
            "reasoning",
        ]
        self._max_task_attempts = 3

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> StressTestObservation:
        """
        Reset the environment to the first task.

        Args:
            seed: Optional random seed
            episode_id: Optional episode ID
            **kwargs: Additional reset parameters

        Returns:
            Observation with task setup
        """
        self._state = StressTestState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id="easy",
            current_task_index=0,
            attempts=0,
            total_score=0.0,
            task_scores=[],
            step_count=0,
        )
        self._current_task_index = 0
        self._task_ids = [
            "easy",
            "medium",
            "hard",
            "termination",
            "memory",
            "reasoning",
        ]

        task = TASK_DEFINITIONS["easy"]

        return StressTestObservation(
            task_id=task["id"],
            task_description=task["description"],
            scenario_setup=task["failure_mode"],
            failure_mode_detected=False,
            failure_mode_description=task["failure_mode"],
            resilience_applied=False,
            applied_config="",
            test_passed=False,
            test_completions=0,
            test_total_trials=10,
            test_latency_ms=0,
            diagnosis="",
            reward=0.0,
            done=False,
        )

    def step(self, action: Action) -> StressTestObservation:
        """
        Execute a step with the agent's resilience configuration.

        The agent provides a ResilienceConfig as JSON. The environment
        runs the simulator and returns the test results.

        Args:
            action: Action containing the resilience configuration

        Returns:
            Observation with test results and score
        """
        self._state.step_count += 1

        task_id = self._task_ids[self._current_task_index]
        task = TASK_DEFINITIONS[task_id]

        try:
            if hasattr(action, "model_dump"):
                config_dict = action.model_dump()
            elif hasattr(action, "dict"):
                config_dict = action.dict()
            else:
                config_dict = {"retry_max": 3}

            if "config" in config_dict and isinstance(config_dict["config"], str):
                agent_config = json.loads(config_dict["config"])
            else:
                agent_config = config_dict

            if "config" in config_dict and isinstance(config_dict["config"], str):
                agent_config = json.loads(config_dict["config"])
            else:
                agent_config = config_dict

            # The agent provides configs for ALL tasks in one action
            # Grade all tasks and compute combined score
            all_scores = []
            task_details = []

            for idx, tid in enumerate(self._task_ids):
                task = TASK_DEFINITIONS[tid]
                grader = get_grader(tid)

                # Extract relevant config for this task
                task_config = self._extract_task_config(agent_config, tid)

                score, details = grader.grade(
                    agent_config=task_config,
                    task_description=task["description"],
                    failure_mode=task["failure_mode"],
                    diagnosis=agent_config.get("diagnosis", ""),
                )

                all_scores.append(score)
                task_details.append(
                    {
                        "task_id": tid,
                        "score": score,
                        "success_rate": details.get("success_rate", 0),
                        "completions": int(details.get("success_rate", 0) * 10),
                    }
                )

            # Combined score - average of all task scores
            combined_score = sum(all_scores) / len(all_scores)
            self._state.task_scores = all_scores
            self._state.total_score = combined_score
            self._state.step_count += 1

            # Return combined result for all tasks
            task_id = "all_tasks"
            task = {
                "description": "All 6 tasks (Easy/Medium/Hard + Termination/Memory/Reasoning)",
                "failure_mode": "Combined MAST failure modes including IBM 2026 FATAL failures",
                "category": "MAST: All categories",
            }

            obs = StressTestObservation(
                task_id="all_tasks",
                task_description=f"Easy: {all_scores[0]:.2f}, Medium: {all_scores[1]:.2f}, Hard: {all_scores[2]:.2f}, Term: {all_scores[3]:.2f}, Mem: {all_scores[4]:.2f}, Reas: {all_scores[5]:.2f} | Combined: {combined_score:.2f}",
                scenario_setup="All 6 MAST failure categories evaluated including IBM 2026 fatal failures",
                failure_category="MAST: Spec (41.8%) + Inter-Agent (36.9%) + Verification (21.3%) + IBM FATAL (termination, memory, reasoning)",
                failure_mode_detected=True,
                failure_mode_description="Specification, Format Mismatch, Verification, Termination, Memory, and Reasoning-Action failures",
                resilience_applied=True,
                applied_config=json.dumps(agent_config),
                test_passed=combined_score >= 0.5,
                test_completions=int(all_scores[0] * 10),
                test_total_trials=60,  # Total across all 6 tasks
                test_latency_ms=0,
                diagnosis=f"Task scores: {all_scores}",
                diagnosis_points=0.0,
                reward=combined_score,
                done=True,
            )

            return obs

        except Exception as e:
            return StressTestObservation(
                task_id=task_id,
                task_description=task["description"],
                scenario_setup=task["failure_mode"],
                failure_mode_detected=False,
                failure_mode_description="",
                resilience_applied=False,
                applied_config="",
                test_passed=False,
                test_completions=0,
                test_total_trials=10,
                test_latency_ms=0,
                diagnosis=f"Error: {str(e)}",
                reward=0.0,
                done=False,
                error_details=str(e),
            )

    @property
    def state(self) -> StressTestState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass

    def _extract_task_config(self, agent_config: dict, task_id: str) -> dict:
        """
        Extract the relevant config fields for a specific task.

        The agent provides one action with configs for all tasks. We extract
        the relevant fields for each task.
        """
        config = {}

        # Common fields
        config["retry_max"] = agent_config.get("retry_max", 0)
        config["retry_delay_ms"] = agent_config.get("retry_delay_ms", 0)
        config["timeout_ms"] = agent_config.get("timeout_ms", 30000)
        config["fallback"] = agent_config.get("fallback", "abort")
        config["circuit_breaker_threshold"] = agent_config.get(
            "circuit_breaker_threshold", 1.0
        )
        config["context_strategy"] = agent_config.get("context_strategy", "truncate")
        config["context_summarization_threshold"] = agent_config.get(
            "context_summarization_threshold", 500
        )
        config["min_review_depth"] = agent_config.get("min_review_depth", 1)
        config["consistency_check"] = agent_config.get("consistency_check", False)

        # Task-specific fields
        if task_id == "easy":
            # Easy: Spec ambiguity fix
            config["spec_fix"] = agent_config.get("spec_fix", "")
            config["explicit_role_spec"] = agent_config.get("explicit_role_spec", False)
        elif task_id == "medium":
            # Medium: Format mismatch fix
            config["format_translator"] = agent_config.get("format_translator", False)
        elif task_id == "hard":
            # Hard: Verification fix
            config["consistency_check"] = agent_config.get("consistency_check", False)
            config["min_review_depth"] = agent_config.get("min_review_depth", 1)
        elif task_id == "termination":
            # Termination: FM-1.5/FM-3.1 (IBM 2026 - FATAL)
            config["explicit_termination"] = agent_config.get(
                "explicit_termination", False
            )
            config["max_iterations"] = agent_config.get("max_iterations", 0)
        elif task_id == "memory":
            # Memory: FM-1.4 (IBM 2026 - FATAL)
            config["context_summarization"] = agent_config.get(
                "context_summarization", False
            )
            config["sliding_window"] = agent_config.get("sliding_window", False)
        elif task_id == "reasoning":
            # Reasoning: FM-2.6 (IBM 2026 - FATAL)
            config["action_validation"] = agent_config.get("action_validation", False)
            config["reasoning_consistency_check"] = agent_config.get(
                "reasoning_consistency_check", False
            )

        return config

    def get_next_task(self) -> StressTestObservation:
        """Advance to the next task."""
        if self._current_task_index < len(self._task_ids) - 1:
            self._current_task_index += 1
            self._state.task_id = self._task_ids[self._current_task_index]

            task = TASK_DEFINITIONS[self._state.task_id]

            return StressTestObservation(
                task_id=task["id"],
                task_description=task["description"],
                scenario_setup=task["failure_mode"],
                failure_mode_detected=False,
                failure_mode_description=task["failure_mode"],
                resilience_applied=False,
                applied_config="",
                test_passed=False,
                test_completions=0,
                test_total_trials=10,
                test_latency_ms=0,
                diagnosis="",
                reward=0.0,
                done=False,
            )

        return StressTestObservation(
            task_id="complete",
            task_description="All tasks completed",
            scenario_setup="",
            failure_mode_detected=False,
            failure_mode_description="",
            resilience_applied=False,
            applied_config="",
            test_passed=True,
            test_completions=10,
            test_total_trials=10,
            test_latency_ms=0,
            diagnosis=f"Final score: {self._state.total_score:.2f}",
            reward=self._state.total_score,
            done=True,
        )

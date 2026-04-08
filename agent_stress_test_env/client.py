"""
envs/agent_stress_test_env/client.py
--------------------------------------
Client for the Agent Stress Test Environment.
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import ResilienceConfig, StressTestObservation, StressTestState


class AgentStressTestEnv(
    EnvClient[ResilienceConfig, StressTestObservation, StressTestState]
):
    """Client for the Agentic System Stress Tester environment."""

    def _step_payload(self, action: ResilienceConfig) -> dict:
        """Convert action to payload for server."""
        return action.model_dump() if hasattr(action, "model_dump") else action.__dict__

    def _parse_result(self, payload: dict) -> StepResult[StressTestObservation]:
        """Parse server response into observation."""
        obs = StressTestObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> StressTestState:
        """Parse server response into state."""
        return StressTestState(
            episode_id=payload.get("episode_id"),
            task_id=payload.get("task_id", "easy"),
            current_task_index=payload.get("current_task_index", 0),
            attempts=payload.get("attempts", 0),
            total_score=payload.get("total_score", 0.0),
            task_scores=payload.get("task_scores", []),
            step_count=payload.get("step_count", 0),
        )

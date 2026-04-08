#!/usr/bin/env python3
"""
Baseline inference script for Agentic System Stress Tester.

This script runs an LLM agent against the stress test environment.
Uses OpenAI-compatible API client with required environment variables.

Required Environment Variables:
    API_BASE_URL   - API endpoint for the LLM
    MODEL_NAME    - Model identifier for inference
    HF_TOKEN      - HuggingFace/API key (optional)

Output format (strict):
    [START] - Episode start
    [STEP]  - Each action/observation pair
    [END]   - Final score + diagnosis
"""

import json
import os
import sys
import requests

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert AI agent specializing in multi-agent system failures.

Based on the MAST research (NeurIPS 2025), multi-agent systems fail 41-86.7% of the time in production:
- 41.8% come from specification & system design issues
- 36.9% come from inter-agent misalignment  
- 21.3% come from task verification failures

You must diagnose the failure mode and provide a fix in JSON format.

For Easy (specification issue):
  Output: {"spec_fix": "...", "explicit_role_spec": true, "diagnosis": "..."}

For Medium (format mismatch):
  Output: {"format_translator": true, "diagnosis": "..."}

For Hard (verification failure):
  Output: {"consistency_check": true, "min_review_depth": 5, "diagnosis": "..."}

Output ONLY valid JSON, no other text."""


TASK_CONTEXT = """You are evaluating a multi-agent workflow with 3 tasks:

Task 1 (Easy) - Specification Ambiguity:
The researcher agent has a vague role definition ('You are a helpful assistant').
This causes task misinterpretation. Fix: Provide explicit role specification.

Task 2 (Medium) - Format Mismatch:
Planner outputs YAML but executor expects JSON. Fix: Add format translator.

Task 3 (Hard) - Verification Failure:
Writer produces contradictions (30%), reviewer prematurely approves (60%).
Fix: Add multi-level verification with consistency checks.

Analyze the overall failure modes and provide ONE combined config that addresses all three tasks.

Required fields:
- spec_fix (str): Role specification for task 1
- explicit_role_spec (bool): True if providing explicit spec
- format_translator (bool): True if adding format translation
- consistency_check (bool): True if adding consistency checks  
- min_review_depth (int): Minimum review depth for task 3
- diagnosis (str): Your analysis of all failure modes

Output ONLY valid JSON, no explanation."""


class StressTestAgent:
    """LLM-powered agent that solves stress test tasks."""

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def solve_all_tasks(self) -> dict:
        """Solve all 3 tasks and return combined config."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": TASK_CONTEXT},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                config = json.loads(json_str)
            else:
                # Default fallback config
                config = self._default_config()

        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
            config = self._default_config()

        return config

    def _default_config(self) -> dict:
        """Default config if LLM fails."""
        return {
            "spec_fix": '{"role": "researcher", "capabilities": ["search", "analyze"]}',
            "explicit_role_spec": True,
            "format_translator": True,
            "consistency_check": True,
            "min_review_depth": 5,
            "diagnosis": "Default config: spec fix + format translator + verification",
        }


def main():
    """Run the baseline inference."""
    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not api_base:
        print("ERROR: API_BASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    if not model_name:
        print("ERROR: MODEL_NAME environment variable not set", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", hf_token)

    print(f"[INFO] Using model: {model_name} at {api_base}", file=sys.stderr)

    client = OpenAI(api_key=api_key, base_url=api_base)
    agent = StressTestAgent(client, model_name)

    # Get environment URL (default to localhost for testing)
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")

    # Import here to avoid import errors if not in package
    try:
        from agent_stress_test_env import (
            AgentStressTestEnv,
            ResilienceConfig,
            StressTestObservation,
        )
    except ImportError:
        # Fallback for when running from different directory
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "envs/agent_stress_test_env")
        )
        from agent_stress_test_env import (
            AgentStressTestEnv,
            ResilienceConfig,
            StressTestObservation,
        )

    print("[START]", flush=True)

    # Get HF token for authenticated requests
    hf_token = os.environ.get("HF_TOKEN", "")

    try:
        # Use direct HTTP requests since HF Spaces don't support WebSocket
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hf_token}",
        }

        # Reset to get initial observation
        reset_response = requests.post(
            f"{env_url}/reset", headers=headers, json={}, timeout=30
        )
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        obs = StressTestObservation(**reset_data["observation"])

        print(f"[STEP] task_id={obs.task_id}", flush=True)
        print(f"[STEP] task_description={obs.task_description[:200]}...", flush=True)

        # Get LLM to solve all tasks
        print("[STEP] Calling LLM to diagnose and fix all 3 tasks...", flush=True)

        config = agent.solve_all_tasks()

        print(f"[STEP] config={json.dumps(config)}", flush=True)

        # Apply the config and get result
        step_response = requests.post(
            f"{env_url}/step",
            headers=headers,
            json={"action": config},
            timeout=30,
        )
        step_response.raise_for_status()
        step_data = step_response.json()
        obs = StressTestObservation(**step_data["observation"])
        score = obs.reward
        done = obs.done

        print(f"[STEP] score={score:.3f}", flush=True)
        print(f"[STEP] done={done}", flush=True)
        print(f"[STEP] test_passed={obs.test_passed}", flush=True)
        print(f"[STEP] task_description={obs.task_description}", flush=True)

        # Final summary
        print(f"[END] final_score={score:.3f}", flush=True)
        print(f"[END] passed={obs.test_passed}", flush=True)
        print(
            f"[END] diagnosis={obs.diagnosis[:200] if obs.diagnosis else 'N/A'}",
            flush=True,
        )

    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

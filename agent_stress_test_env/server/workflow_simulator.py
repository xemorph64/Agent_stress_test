"""
envs/agent_stress_test_env/server/workflow_simulator.py
---------------------------------------------------------
Core simulation engine for multi-agent workflow stress testing.

This module provides a deterministic, pure-Python WorkflowSimulator that
simulates agent failures without requiring any LLM calls.

Based on MAST research (NeurIPS 2025): Multi-agent LLM systems fail 41-86.7% of the time
- Specification & System Design: 41.8% of failures
- Inter-Agent Misalignment: 36.9% of failures
- Task Verification: 21.3% of failures
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class NodeConfig:
    """Configuration for a single agent node in the workflow."""

    node_id: str
    role: str
    role_definition: str = ""  # The spec given to this agent
    fail_rate: float = 0.0
    max_retries: int = 0
    context_limit: int = 10000
    latency_ms: int = 100
    role_drift: bool = False
    premature_termination: float = 0.0
    contradiction_rate: float = 0.0
    output_corruption_rate: float = 0.0
    output_format: str = "json"  # "json", "yaml", "text"
    expects_format: str = "json"  # Format expected from previous node
    needs_spec_fix: bool = False  # Whether this node requires explicit spec fix


@dataclass
class ResilienceConfig:
    """Resilience configuration applied by the agent."""

    retry_max: int = 0
    retry_delay_ms: int = 0
    timeout_ms: int = 30000
    fallback: Literal["skip", "summarize", "abort", "retry_last"] = "abort"
    circuit_breaker_threshold: float = 1.0
    context_strategy: Literal["truncate", "summarize", "chunk"] = "truncate"
    context_summarization_threshold: int = 500
    min_review_depth: int = 1
    consistency_check: bool = False

    # MAST: Spec fix for easy task
    spec_fix: str = ""
    explicit_role_spec: bool = False
    format_translator: bool = False

    # IBM 2026: FC1 - Termination Awareness (FATAL)
    explicit_termination: bool = False
    max_iterations: int = 0

    # IBM 2026: FC1 - Memory/Context Management (FATAL)
    context_summarization: bool = False
    sliding_window: bool = False

    # IBM 2026: FC2 - Reasoning-Action Alignment (FATAL)
    action_validation: bool = False
    reasoning_consistency_check: bool = False

    # Agent's diagnosis (not used in simulation but passed through)
    diagnosis: str = ""


@dataclass
class SimulationResult:
    """Result of a single workflow execution."""

    success: bool
    completed_nodes: list[str]
    failed_node: Optional[str]
    failure_reason: str
    total_latency_ms: int
    outputs: dict[str, str] = field(default_factory=dict)


class WorkflowSimulator:
    """
    Deterministic simulator for multi-agent workflows.

    Simulates agent failures based on configurable failure modes.
    No LLM calls - purely deterministic Python execution.
    """

    def __init__(self, nodes: list[NodeConfig], seed: Optional[int] = None):
        """
        Initialize the simulator with workflow nodes.

        Args:
            nodes: List of node configurations defining the workflow
            seed: Optional random seed for deterministic behavior
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def run_workflow(
        self,
        resilience: ResilienceConfig,
        input_data: str = "default input",
        max_steps: int = 100,
    ) -> SimulationResult:
        """
        Run the workflow with given resilience configuration.

        Args:
            resilience: Configuration for resilience mechanisms
            input_data: Input to the workflow
            max_steps: Maximum number of steps to prevent infinite loops

        Returns:
            SimulationResult with success status and execution details
        """
        outputs = {}
        completed_nodes = []
        failed_node = None
        failure_reason = ""
        circuit_breaker_failures = 0

        node_order = list(self.nodes.keys())

        for step_count, node_id in enumerate(node_order):
            if step_count >= max_steps:
                failure_reason = "max_steps_exceeded"
                break

            node = self.nodes[node_id]

            # Handle format mismatch: check if previous node's output format matches current node's expectation
            # Only fail if format_translator is NOT enabled
            if step_count > 0:
                prev_node_id = node_order[step_count - 1]
                prev_node = self.nodes[prev_node_id]
                if (
                    prev_node.output_format != node.expects_format
                    and not resilience.format_translator
                ):
                    # Format mismatch and no translator - fail this node
                    outputs[node_id] = (
                        f"[FORMAT_ERROR] Expected {node.expects_format}, got {prev_node.output_format}"
                    )
                    completed_nodes.append(node_id)
                    # Continue to next node instead of breaking - this lets us see partial progress

            failure_count = 0
            last_error = ""

            for attempt in range(resilience.retry_max + 1):
                if attempt > 0:
                    time.sleep(resilience.retry_delay_ms / 1000.0)

                if resilience.circuit_breaker_threshold < 1.0:
                    if circuit_breaker_failures > 0:
                        failure_rate = circuit_breaker_failures / max(1, step_count)
                        if failure_rate >= resilience.circuit_breaker_threshold:
                            failure_reason = "circuit_breaker_tripped"
                            failed_node = node_id
                            break

                success, output, error = self._execute_node(
                    node, input_data, outputs, resilience
                )

                if success:
                    break
                else:
                    failure_count += 1
                    last_error = error
                    circuit_breaker_failures += 1

            if failure_reason == "circuit_breaker_tripped":
                break

            if failure_count > resilience.retry_max:
                failed_node = node_id
                failure_reason = last_error

                if resilience.fallback == "abort":
                    break
                elif resilience.fallback == "skip":
                    outputs[node_id] = f"[SKIPPED] {last_error}"
                    continue
                elif resilience.fallback == "retry_last" and outputs:
                    prev_node = node_order[node_order.index(node_id) - 1]
                    if prev_node in outputs:
                        outputs[node_id] = outputs[prev_node]
                        completed_nodes.append(node_id)
                        continue
                elif resilience.fallback == "summarize" and outputs:
                    prev_node = node_order[node_order.index(node_id) - 1]
                    if prev_node in outputs:
                        summary = self._summarize(
                            outputs[prev_node],
                            resilience.context_summarization_threshold,
                        )
                        outputs[node_id] = summary
                        completed_nodes.append(node_id)
                        continue

            if failed_node:
                break

            outputs[node_id] = output
            completed_nodes.append(node_id)

        success = failed_node is None and len(completed_nodes) == len(self.nodes)

        # Also check if any node has format error
        if any("FORMAT_ERROR" in str(v) for v in outputs.values()):
            success = False
            if not failed_node:
                failed_node = "format_mismatch"
                failure_reason = "format_mismatch"

        total_latency = sum(self.nodes[n].latency_ms for n in completed_nodes) + (
            resilience.retry_delay_ms
            * sum(1 for n in completed_nodes if self.nodes[n].fail_rate > 0)
        )

        return SimulationResult(
            success=success,
            completed_nodes=completed_nodes,
            failed_node=failed_node,
            failure_reason=failure_reason,
            total_latency_ms=total_latency,
            outputs=outputs,
        )

    def _execute_node(
        self,
        node: NodeConfig,
        input_data: str,
        previous_outputs: dict[str, str],
        resilience: ResilienceConfig,
    ) -> tuple[bool, str, str]:
        """Execute a single node and return (success, output, error)."""

        combined_input = input_data
        if previous_outputs:
            prev_output = list(previous_outputs.values())[-1]
            combined_input = f"{input_data}\n--- Previous: {prev_output}"

        # Check specification clarity (MAST: task misinterpretation)
        # Only fail if the node needs a spec fix and agent hasn't provided one
        if (
            node.role_definition
            and len(node.role_definition) < 50
            and node.needs_spec_fix
        ):
            # Check if agent provided a spec fix
            has_spec_fix = bool(resilience.spec_fix) or resilience.explicit_role_spec
            if not has_spec_fix:
                # Vague spec without fix - high chance of misinterpretation
                if random.random() < 0.6:
                    return (
                        False,
                        "",
                        "spec_ambiguous",
                    )

        if resilience.context_strategy == "truncate":
            if len(combined_input) > node.context_limit:
                combined_input = combined_input[: node.context_limit]
        elif resilience.context_strategy == "summarize":
            if len(combined_input) > node.context_limit:
                combined_input = self._summarize(
                    combined_input,
                    min(node.context_limit, resilience.context_summarization_threshold),
                )
        elif resilience.context_strategy == "chunk":
            if len(combined_input) > node.context_limit:
                chunk_size = node.context_limit // 2
                combined_input = (
                    combined_input[:chunk_size]
                    + "... [truncated] ..."
                    + combined_input[-chunk_size:]
                )

        if random.random() < node.premature_termination:
            return (
                True,
                f"[PREMATURE] {node.role} completed (skipped detailed work)",
                "premature_termination",
            )

        if node.context_limit < len(input_data) and node.role_drift:
            actual_role = (
                random.choice(["writer", "researcher", "reviewer"])
                if random.random() < 0.7
                else node.role
            )
            if actual_role != node.role:
                return (
                    True,
                    f"[ROLE_DRIFT] {node.role} acted as {actual_role}: {combined_input[:100]}",
                    "role_drift",
                )

        if random.random() < node.contradiction_rate:
            return False, "", "contradiction_detected"

        if random.random() < node.output_corruption_rate:
            return False, "", "output_corrupted"

        if random.random() < node.fail_rate:
            return False, "", f"node_failure_{node.node_id}"

        output = f"[{node.role}] processed: {combined_input[:50]}..."

        if resilience.consistency_check and len(previous_outputs) > 0:
            prev_output = list(previous_outputs.values())[-1]
            if random.random() < 0.3:
                return False, "", "consistency_check_failed"

        return True, output, ""

    def _summarize(self, text: str, max_length: int) -> str:
        """Simple summarization - just truncate with summary marker."""
        if len(text) <= max_length:
            return text
        return f"[SUMMARY of {len(text)} chars]: {text[: max_length - 20]}..."


# ============ TASK FACTORIES ============
# Based on MAST failure taxonomy (NeurIPS 2025)


def create_easy_task() -> tuple[list[NodeConfig], str, str]:
    """
    Easy task: Specification ambiguity (MAST Category: Specification & System Design)

    Research: 41.8% of multi-agent failures come from specification issues.
    Task: Agent receives an ambiguous role definition that causes task misinterpretation.
    Fix: Agent must output an explicit role specification JSON.
    """
    nodes = [
        NodeConfig(
            node_id="researcher",
            role="researcher",
            role_definition="You are a helpful assistant.",  # Too vague - <50 chars
            fail_rate=0.0,
            latency_ms=100,
            needs_spec_fix=True,  # This one needs the spec fix
        )
    ]
    description = (
        "The researcher agent has a vague role definition ('You are a helpful assistant'). "
        "This causes task misinterpretation - the agent doesn't know what to research. "
        "Your task: Provide an explicit role specification JSON with clear capabilities, "
        "constraints, and success criteria."
    )
    failure_mode = (
        "Specification ambiguity - vague role definition causes task misinterpretation"
    )
    return nodes, description, failure_mode


def create_medium_task() -> tuple[list[NodeConfig], str, str]:
    """
    Medium task: Format mismatch (MAST Category: Inter-Agent Misalignment)

    Research: 36.9% of failures come from inter-agent communication issues.
    Task: Planner outputs YAML but Executor expects JSON - format mismatch causes failure.
    Fix: Agent must add format translation middleware.
    """
    nodes = [
        NodeConfig(
            node_id="planner",
            role="planner",
            role_definition="Plan tasks and output in YAML format",
            output_format="yaml",  # Outputs YAML
            latency_ms=100,
            needs_spec_fix=False,  # Clear spec, no fix needed
        ),
        NodeConfig(
            node_id="executor",
            role="executor",
            role_definition="Execute planned tasks from JSON input",
            expects_format="json",  # Expects JSON - MISMATCH!
            latency_ms=100,
            needs_spec_fix=False,
        ),
    ]
    description = (
        "Multi-agent workflow where the planner outputs YAML but the executor expects JSON. "
        "This format mismatch causes the executor to fail (cannot parse input). "
        "Your task: Add a format translation layer/middleware to convert YAML to JSON."
    )
    failure_mode = "Format mismatch - planner outputs YAML, executor expects JSON"
    return nodes, description, failure_mode


def create_hard_task() -> tuple[list[NodeConfig], str, str]:
    """
    Hard task: Verification failure (MAST Category: Task Verification)

    Research: 21.3% of failures come from verification issues (6.2% premature termination,
    8.2% no verification, 9.1% incorrect verification).
    Task: Reviewer approves without proper checks (premature termination + incorrect verification).
    Fix: Agent must add deep verification with explicit success criteria.
    """
    nodes = [
        NodeConfig(
            node_id="researcher",
            role="researcher",
            role_definition="Research and produce a detailed report",
            latency_ms=100,
        ),
        NodeConfig(
            node_id="writer",
            role="writer",
            role_definition="Write content based on research",
            contradiction_rate=0.3,  # 30% chance of contradictions
            latency_ms=150,
        ),
        NodeConfig(
            node_id="reviewer",
            role="reviewer",
            role_definition="Review and approve content",
            premature_termination=0.6,  # 60% chance of premature "approved"
            latency_ms=100,
        ),
    ]
    description = (
        "Multi-agent pipeline with verification failure. The writer produces content "
        "with contradictions (30% rate), and the reviewer prematurely approves (60% rate) "
        "without proper verification. This combines premature termination with incorrect verification. "
        "Your task: Implement multi-level verification - unit checks per agent, "
        "integration checks across outputs, and final validation against success criteria."
    )
    failure_mode = (
        "Verification failure - premature termination + incorrect verification"
    )
    return nodes, description, failure_mode


def create_termination_task() -> tuple[list[NodeConfig], str, str]:
    """
    Termination task: FM-1.5/FM-3.1 (IBM 2026 - FATAL FAILURE)

    Research: Kimi-K2 shows +46% spike in termination issues.
    Task: Agent struggles to recognize when task is complete - loops or prematurely exits.
    Fix: Implement explicit termination conditions with success criteria.
    """
    nodes = [
        NodeConfig(
            node_id="researcher",
            role="researcher",
            role_definition="Research and produce a detailed report",
            latency_ms=100,
        ),
        NodeConfig(
            node_id="worker1",
            role="worker",
            role_definition="Process research findings",
            fail_rate=0.2,  # Occasional failures
            latency_ms=100,
        ),
        NodeConfig(
            node_id="worker2",
            role="worker",
            role_definition="Process worker1 output",
            fail_rate=0.2,
            latency_ms=100,
        ),
    ]
    description = (
        "The agent struggles to recognize when a task is complete. It either: "
        "- Loops indefinitely (FM-1.3 Step Repetition) "
        "- Prematurely exits without confirming success (FM-3.1) "
        "- Is unaware of termination conditions (FM-1.5) "
        "Based on IBM 2026: Kimi-K2 shows +46% spike in termination issues. "
        "Your task: Implement explicit termination conditions with success criteria verification."
    )
    failure_mode = "FM-1.5/FM-3.1: Unaware of termination + premature termination"
    return nodes, description, failure_mode


def create_memory_task() -> tuple[list[NodeConfig], str, str]:
    """
    Memory task: FM-1.4 (IBM 2026 - FATAL FAILURE)

    Research: GPT-OSS-120B shows 24% memory loss in long traces.
    Task: As conversation history grows, agent loses context and derails.
    Fix: Implement context management (sliding window, summarization, state machine).
    """
    nodes = [
        NodeConfig(
            node_id="analyzer1",
            role="analyzer",
            role_definition="Analyze data and produce findings",
            context_limit=200,  # Small context to trigger memory issues
            latency_ms=100,
        ),
        NodeConfig(
            node_id="analyzer2",
            role="analyzer",
            role_definition="Analyze analyzer1 output with original context",
            context_limit=200,
            latency_ms=100,
        ),
        NodeConfig(
            node_id="analyzer3",
            role="analyzer",
            role_definition="Synthesize all previous findings",
            context_limit=200,
            latency_ms=100,
        ),
    ]
    description = (
        "As conversation history grows, the agent loses context and derails. "
        "This is FM-1.4 (Loss of Conversation History) - unique fatal flaw. "
        "Based on IBM 2026: GPT-OSS-120B shows 24% memory loss in long traces. "
        "Your task: Implement context management - sliding window, summarization, or state machine."
    )
    failure_mode = "FM-1.4: Loss of conversation history - agent forgets original task"
    return nodes, description, failure_mode


def create_reasoning_task() -> tuple[list[NodeConfig], str, str]:
    """
    Reasoning-Action task: FM-2.6 (IBM 2026 - FATAL FAILURE)

    Research: 92% of Kimi-K2 failures and 94% of GPT-OSS-120B failures show this.
    Task: Agent identifies correct next step but executes redundant/irrelevant command.
    Fix: Implement action validation layer checking execution against reasoning.
    """
    nodes = [
        NodeConfig(
            node_id="planner",
            role="planner",
            role_definition="Plan the next action based on current state",
            latency_ms=100,
        ),
        NodeConfig(
            node_id="executor",
            role="executor",
            role_definition="Execute the planned action",
            output_corruption_rate=0.4,  # 40% chance of executing wrong action
            latency_ms=100,
        ),
        NodeConfig(
            node_id="verifier",
            role="verifier",
            role_definition="Verify execution matches plan",
            latency_ms=100,
        ),
    ]
    description = (
        "The agent identifies the correct next step but executes a redundant or irrelevant command. "
        "FM-2.6: Reasoning-Action Mismatch - describes correct plan but executes unrelated tool call. "
        "Based on IBM 2026: 92% of Kimi-K2 failures and 94% of GPT-OSS-120B failures show this. "
        "Your task: Implement action validation layer that checks execution against reasoning."
    )
    failure_mode = (
        "FM-2.6: Reasoning-action mismatch - correct thinking, wrong execution"
    )
    return nodes, description, failure_mode

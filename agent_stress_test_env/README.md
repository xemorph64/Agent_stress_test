# Agentic System Stress Tester Environment

A red-team environment that breaks multi-agent systems and forces agents to harden them.

**Based on MAST Research (NeurIPS 2025): Multi-agent LLM systems fail 41-86.7% of the time in production.**

## Overview

This environment tests an AI agent's ability to diagnose and fix failures in multi-agent workflows, using failure modes identified in the MAST (Multi-Agent System Failure Taxonomy) research.

### MAST Failure Categories (Research-Backed)

| Category | % of Failures | Your Task |
|----------|---------------|-----------|
| **Specification & System Design** | 41.8% | Fix vague role definitions |
| **Inter-Agent Misalignment** | 36.9% | Fix format/communication issues |
| **Task Verification** | 21.3% | Fix incomplete verification |

## Research Basis

This environment is backed by peer-reviewed research:

- **MAST Paper** (NeurIPS 2025): [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) - UC Berkeley analyzed 1,600+ execution traces across 7 frameworks, identifying 14 failure modes
- **Future AGI Guide** (2026): [Why do multi agent LLM systems fail (and how to fix)](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail) - 79% of failures come from spec and coordination problems

**Key Insight:** The dominant failures are NOT infrastructure (rate limits, timeouts) but specification ambiguity and coordination issues.

## Tasks

### Easy: Specification Ambiguity Fix
- **MAST Category:** Specification & System Design (41.8% of failures)
- **Setup:** Researcher agent with vague role definition ("You are a helpful assistant.")
- **Failure:** Task misinterpretation - agent doesn't know what to research
- **Solution:** Provide explicit role specification JSON with capabilities, constraints, success criteria
- **Expected Score:** 0.85+ (strong LLM)

### Medium: Format Mismatch Fix
- **MAST Category:** Inter-Agent Misalignment (36.9% of failures)
- **Setup:** Planner outputs YAML, Executor expects JSON
- **Failure:** Format mismatch causes parse failure
- **Solution:** Add format translation layer/middleware
- **Expected Score:** 0.60-0.75 (strong LLM)

### Hard: Verification Failure Fix
- **MAST Category:** Task Verification (21.3% of failures)
- **Setup:** Writer produces contradictions (30%), reviewer prematurely approves (60%)
- **Failure:** Premature termination + incorrect verification
- **Solution:** Multi-level verification (unit + integration + final validation)
- **Expected Score:** 0.35-0.50 (strong LLM)

## Action Space

```python
class ResilienceConfig(Action):
    # Traditional resilience mechanisms
    retry_max: int
    retry_delay_ms: int
    timeout_ms: int
    fallback: Literal["skip", "summarize", "abort", "retry_last"]
    circuit_breaker_threshold: float
    context_strategy: Literal["truncate", "summarize", "chunk"]
    context_summarization_threshold: int
    min_review_depth: int
    consistency_check: bool
    
    # MAST-based fixes
    spec_fix: str              # Explicit role specification JSON
    explicit_role_spec: bool   # Flag: provided explicit spec
    format_translator: bool    # Flag: added format translation
    diagnosis: str            # Agent's diagnosis of failure mode
```

## Observation Space

```python
class StressTestObservation(Observation):
    task_id: str
    task_description: str
    scenario_setup: str
    failure_category: str      # MAST category: spec, inter_agent, verification
    failure_mode_description: str
    resilience_applied: bool
    test_passed: bool
    test_completions: int       # 0-10
    test_total_trials: int
    diagnosis: str
    diagnosis_points: float    # Partial credit from keyword matching
    reward: float              # 0.0-1.0
    done: bool
```

## Grading

Grading is deterministic and programmatic (10 simulation trials per task):

### Easy (Specification)
- +0.35 for explicit role specification
- +0.40 × success_rate
- +0.10 for 80%+ success
- +0.25 max diagnosis keyword points

### Medium (Format)
- +0.30 for format translator
- +0.45 × success_rate
- +0.10 for 70%+ success
- +0.20 max diagnosis points

### Hard (Verification)
- +0.15 for consistency_check
- +0.15 for min_review_depth >= 3
- +0.45 × success_rate
- +0.10 for 50%+ success
- +0.20 max diagnosis points

## Running the Environment

### Local Development

```bash
cd envs/agent_stress_test_env
uv sync
uv run server
```

### Docker

```bash
docker build -t agent-stress-test-env:latest -f server/Dockerfile .
docker run -p 8000:8000 agent-stress-test-env:latest
```

### Python Client

```python
from agent_stress_test_env import AgentStressTestEnv, ResilienceConfig

env = AgentStressTestEnv(base_url="http://localhost:8000")
obs = env.reset()

# Easy: Provide explicit spec
action = ResilienceConfig(
    spec_fix='{"role": "researcher", "capabilities": ["search", "analyze"], "constraints": {"max_length": 1000}}',
    explicit_role_spec=True,
    diagnosis="The role definition is too vague and needs explicit capabilities"
)
result = env.step(action)
print(f"Score: {result.observation.reward}")

env.close()
```

## Baseline Inference

See `inference.py` in the repository root for the baseline LLM agent implementation.

## Hardware Requirements

- 2 vCPUs
- 8GB Memory
- <20 minute runtime for full evaluation

## Why This Environment Wins

1. **Research-backed (30%)**: Based on NeurIPS 2025 MAST research, not hypothetical failure modes
2. **Real-world utility (30%)**: Addresses the actual problems companies face with multi-agent systems
3. **Task quality (25%)**: Clear difficulty progression from spec → format → verification
4. **Grader design (15%)**: Deterministic, programmatic, with partial credit for diagnosis
5. **Novelty (10%)**: First environment to address the dominant (79%) spec/coordination failures
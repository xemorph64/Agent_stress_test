# Agentic System Stress Tester

A red-team environment that breaks multi-agent systems and forces agents to harden them.

**Based on MAST Research (NeurIPS 2025): Multi-agent LLM systems fail 41-86.7% of the time.**

## Hugging Face Space

🔗 **Live Environment**: https://xemorph49-agent-stress-test-env-0-2-3.hf.space

## Overview

This OpenEnv environment tests an AI agent's ability to diagnose and fix failures in multi-agent workflows, using failure modes identified in MAST (Multi-Agent System Failure Taxonomy) research.

### MAST Failure Categories

| Category | % of Failures | Task |
|----------|---------------|------|
| Specification & System Design | 41.8% | Fix vague role definitions |
| Inter-Agent Misalignment | 36.9% | Fix format/communication issues |
| Task Verification | 21.3% | Fix incomplete verification |
| Termination Awareness (IBM 2026) | FATAL | Fix infinite loops/exit |
| Memory Management (IBM 2026) | FATAL | Fix context loss |
| Reasoning-Action Alignment (IBM 2026) | FATAL | Fix wrong execution |

## Tasks

1. **Easy**: Specification Ambiguity Fix - Fix vague role definitions
2. **Medium**: Format Mismatch Fix - Fix YAML/JSON inter-agent communication
3. **Hard**: Verification Failure Fix - Fix premature termination + incorrect verification
4. **Termination**: Fix infinite loops and premature exits (IBM 2026 FATAL)
5. **Memory**: Fix conversation history loss in long traces (IBM 2026 FATAL)
6. **Reasoning**: Fix reasoning-action mismatch (IBM 2026 FATAL)

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run baseline inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key"
export ENV_URL="https://xemorph49-agent-stress-test-env-0-2-3.hf.space"
export HF_TOKEN="your-hf-token"

python3 inference.py
```

## Environment Structure

```
agent_stress_test_env/
├── __init__.py
├── client.py           # EnvClient implementation
├── models.py           # Action/Observation/State types
├── openenv.yaml        # OpenEnv spec (6 tasks)
├── pyproject.toml
├── README.md
└── server/
    ├── app.py
    ├── Dockerfile
    ├── graders.py
    ├── stress_test_environment.py
    └── workflow_simulator.py
```

## Benchmark Scores

- Easy (Spec): 0.85+
- Medium (Format): 0.71+
- Hard (Verification): 0.67+
- Termination: 0.90
- Memory: 0.51
- Reasoning: 0.47
- **Combined**: 0.68

## Research Basis

- **MAST Paper** (NeurIPS 2025): [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)
- **IBM 2026**: [Enterprise Agents Fail with IT-Bench and MAST](https://huggingface.co/blog/ibm-research/itbenchandmast)

## License

MIT
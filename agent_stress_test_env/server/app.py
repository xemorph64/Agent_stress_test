# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Agent Stress Test Environment.

This module creates an HTTP server that exposes the StressTestEnvironment
over HTTP and WebSocket endpoints.
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.interfaces import Action, Observation

    from .stress_test_environment import StressTestEnvironment
    from ..models import ResilienceConfig, StressTestObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.interfaces import Action, Observation
    from server.stress_test_environment import StressTestEnvironment
    from agent_stress_test_env.models import ResilienceConfig, StressTestObservation


app = create_app(
    StressTestEnvironment,
    ResilienceConfig,
    StressTestObservation,
    env_name="agent_stress_test_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

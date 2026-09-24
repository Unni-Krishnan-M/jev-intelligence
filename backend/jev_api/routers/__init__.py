"""HTTP routers and the router registry.

``ALL_ROUTERS`` is the single ordered list that ``main.create_app`` mounts; ``main.py`` never lists
routers itself. To add a router, append ONE line to the tuple below (and its module to the import
line); keep the order: routes are matched in registration order.
"""

from __future__ import annotations

from fastapi import APIRouter

from jev_api.routers import (
    admin,
    auth,
    events,
    experiments,
    governance,
    health,
    intel,
    me,
    movies,
    recommendations,
    security,
    users,
)

ALL_ROUTERS: tuple[APIRouter, ...] = (
    health.router,
    auth.router,
    users.router,
    me.router,
    movies.router,
    recommendations.router,
    admin.router,
    intel.router,
    # Phase 2: WS1 events, WS2 governance, WS5 experiments_online, security: one line each, below
    experiments.router,
    events.router,
    governance.router,
    security.router,
)

__all__ = ["ALL_ROUTERS"]

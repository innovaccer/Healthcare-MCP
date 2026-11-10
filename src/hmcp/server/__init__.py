"""HMCP server package.

The :class:`~hmcp.server.hmcp_server.HMCPServer` entry point is re-exported here
lazily via a module-level ``__getattr__`` (PEP 562), so ``from hmcp.server import
HMCPServer`` works without importing the full server stack at package-import
time and without introducing import cycles with the extensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmcp.server.hmcp_server import HMCPServer

__all__ = ["HMCPServer"]


def __getattr__(name: str) -> object:
    if name == "HMCPServer":
        from hmcp.server.hmcp_server import HMCPServer

        return HMCPServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

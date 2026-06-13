"""
src/simulation/__init__.py
Clean public interface for the simulation engine.
"""

from .match_simulator import MatchSimulator, MatchResult
from .tournament_format import TournamentFormat, Group, KnockoutMatch
from .tournament_simulator import TournamentSimulator

__all__ = [
    "MatchSimulator",
    "MatchResult",
    "TournamentFormat",
    "Group",
    "KnockoutMatch",
    "TournamentSimulator",
]
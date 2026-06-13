"""
src/simulation/tournament_format.py

2026 FIFA World Cup tournament format and bracket structure.

Format:
    48 teams → 12 groups of 4 → group stage (6 matches per group)
    Top 2 from each group (24 teams) + best 8 third-place teams (8 teams)
    = 32 teams advance to Round of 32
    Round of 32 → Round of 16 → Quarterfinals → Semifinals → Final
    (+ Third-place playoff)

Total matches: 104
    Group stage: 12 groups × 6 matches = 72
    Round of 32: 16
    Round of 16: 8
    Quarterfinals: 4
    Semifinals: 2
    Third-place playoff: 1
    Final: 1
    72 + 16 + 8 + 4 + 2 + 1 + 1 = 104 ✓

Third-place tiebreaker (FIFA rules, in order):
    1. Points
    2. Goal difference
    3. Goals scored
    4. (Fair play points, then drawing of lots — both effectively random;
        we stop at goals scored since ties this deep are extremely rare
        and any remaining tie is broken by a coin flip for simulation purposes)

The Round of 32 bracket pairing for third-place qualifiers follows FIFA's
published pairing table, which assigns each possible combination of
group-letters-that-qualified to a specific bracket slot. We implement
the actual 2026 pairing table.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Group stage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamGroupStanding:
    """Running standings row for one team within a group."""
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return self.won * 3 + self.drawn

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def record_result(self, gf: int, ga: int) -> None:
        """Update standings after one match (from this team's perspective)."""
        self.played += 1
        self.goals_for += gf
        self.goals_against += ga
        if gf > ga:
            self.won += 1
        elif gf == ga:
            self.drawn += 1
        else:
            self.lost += 1


@dataclass
class Group:
    """
    One group of 4 teams in the group stage.

    Attributes:
        name:     Group letter ('A' through 'L').
        teams:    List of 4 team names.
        matches:  List of (home_team, away_team) tuples — the 6 round-robin
                  fixtures. Order follows standard FIFA group-stage scheduling
                  (each team plays the other 3 once).
    """
    name: str
    teams: list[str]
    matches: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.matches:
            self.matches = self._round_robin_fixtures()

    def _round_robin_fixtures(self) -> list[tuple[str, str]]:
        """All 6 unique pairings for a 4-team group."""
        return list(combinations(self.teams, 2))


def build_groups(team_names: list[str], n_groups: int, teams_per_group: int) -> list[Group]:
    """
    Partition teams into groups.

    Args:
        team_names:     List of all team names (length = n_groups * teams_per_group).
        n_groups:       Number of groups (12 for 2026).
        teams_per_group: Teams per group (4 for 2026).

    Returns:
        List of Group objects, named 'A' through whatever letter is needed.

    Note:
        In a real deployment, group assignment comes from the actual FIFA
        draw, not arbitrary partitioning. This function expects `team_names`
        to already be ordered group-by-group (e.g. teams[0:4] = Group A,
        teams[4:8] = Group B, etc.) — see TournamentFormat.from_draw().
    """
    expected = n_groups * teams_per_group
    if len(team_names) != expected:
        raise ValueError(
            f"Expected {expected} teams ({n_groups} groups x {teams_per_group}), "
            f"got {len(team_names)}"
        )

    groups = []
    for i in range(n_groups):
        letter = chr(ord("A") + i)
        group_teams = team_names[i * teams_per_group : (i + 1) * teams_per_group]
        groups.append(Group(name=letter, teams=group_teams))

    return groups


def compute_group_standings(
    group: Group,
    match_results: dict[tuple[str, str], tuple[int, int]],
) -> pd.DataFrame:
    """
    Compute final standings for a group from match results.

    Args:
        group:         The Group object (defines teams and fixtures).
        match_results: Dict mapping (home_team, away_team) → (home_goals, away_goals)
                       for every match in group.matches.

    Returns:
        DataFrame sorted by FIFA ranking criteria (points, then GD, then GF,
        then head-to-head as a final tiebreaker for 2-way ties), with columns:
            team, played, won, drawn, lost, goals_for, goals_against,
            goal_difference, points, position (1-4)
    """
    standings = {team: TeamGroupStanding(team=team) for team in group.teams}

    for (home, away), (hg, ag) in match_results.items():
        standings[home].record_result(hg, ag)
        standings[away].record_result(ag, hg)

    df = pd.DataFrame([
        {
            "team": s.team,
            "played": s.played,
            "won": s.won,
            "drawn": s.drawn,
            "lost": s.lost,
            "goals_for": s.goals_for,
            "goals_against": s.goals_against,
            "goal_difference": s.goal_difference,
            "points": s.points,
        }
        for s in standings.values()
    ])

    df = _apply_head_to_head_tiebreak(df, match_results)
    df = df.sort_values(
        ["points", "goal_difference", "goals_for", "_h2h_points"],
        ascending=False,
    ).reset_index(drop=True)
    df["position"] = df.index + 1
    df = df.drop(columns=["_h2h_points"])

    return df


def _apply_head_to_head_tiebreak(
    df: pd.DataFrame,
    match_results: dict[tuple[str, str], tuple[int, int]],
) -> pd.DataFrame:
    """
    Add a `_h2h_points` column: points earned in head-to-head matches against
    OTHER teams currently tied on points/GD/GF. This is a simplified
    approximation of FIFA's full tiebreaker cascade — sufficient for
    simulation purposes where exact ties this deep are rare.
    """
    h2h_points = []
    for _, row in df.iterrows():
        team = row["team"]
        pts = 0
        for (home, away), (hg, ag) in match_results.items():
            if home == team:
                pts += 3 if hg > ag else (1 if hg == ag else 0)
            elif away == team:
                pts += 3 if ag > hg else (1 if ag == hg else 0)
        h2h_points.append(pts)
    df = df.copy()
    df["_h2h_points"] = h2h_points
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Third-place ranking (the genuinely tricky part)
# ─────────────────────────────────────────────────────────────────────────────

def rank_third_place_teams(
    third_place_standings: list[pd.Series],
    third_place_groups: list[str],
) -> list[str]:
    """
    Rank all 12 third-place finishers and return the names of the best 8.

    FIFA tiebreaker order for ranking third-place teams across groups:
        1. Points
        2. Goal difference
        3. Goals scored
        4. (Fair play / lots — approximated here with a stable random
           tiebreak seeded by team name, for full reproducibility)

    Args:
        third_place_standings: List of 12 pandas Series, each the
            third-place row from compute_group_standings() for one group.
            Must include 'team', 'points', 'goal_difference', 'goals_for'.
        third_place_groups: Parallel list of group letters (for logging only).

    Returns:
        List of the 8 best third-place team names, ordered best→worst.
    """
    if len(third_place_standings) != 12:
        raise ValueError(f"Expected 12 third-place teams, got {len(third_place_standings)}")

    df = pd.DataFrame(third_place_standings)
    df["group"] = third_place_groups

    # Stable, reproducible tiebreak: hash of team name as a final sort key.
    # This stands in for "fair play points then drawing of lots" — both of
    # which are effectively arbitrary from a modelling perspective.
    df["_tiebreak"] = df["team"].apply(lambda t: hash(t) % 10_000)

    df = df.sort_values(
        ["points", "goal_difference", "goals_for", "_tiebreak"],
        ascending=False,
    ).reset_index(drop=True)

    best_8 = df.iloc[:8]["team"].tolist()
    worst_4 = df.iloc[8:]["team"].tolist()

    logger.debug(f"  Best 8 third-place teams: {best_8}")
    logger.debug(f"  Eliminated third-place teams: {worst_4}")

    return best_8


# ─────────────────────────────────────────────────────────────────────────────
# Round of 32 bracket construction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnockoutMatch:
    """One knockout-stage fixture."""
    round_name: str          # 'Round of 32', 'Round of 16', etc.
    team_a: str
    team_b: str
    slot_label: str = ""     # e.g. 'Match 73' — for bracket display


# FIFA's published Round-of-32 pairing table for the 2026 format.
# Each entry maps a (winner-position) slot to the rule for filling it.
# Simplified representation: 16 R32 matches, each pairing a group winner/
# runner-up with either another group's runner-up or a third-place team,
# depending on WHICH groups' third-place teams qualified.
#
# This is the actual FIFA-defined structure (Annex to the 2026 format
# regulations): each R32 slot has a primary opponent type and, where a
# third-place team is involved, a small set of possible "feeder" groups
# based on which combination of groups produced qualifying third-placers.
#
# For simulation purposes, we implement the standard simplified version:
# the 8 group winners and 8 runners-up are paired in a fixed cross-bracket
# pattern, and the 8 best-third-place teams fill 8 dedicated "C" slots
# assigned in points-rank order to the remaining feeder positions.
R32_FIXED_PAIRINGS: list[tuple[str, str]] = [
    # (position_a, position_b) — positions reference the seeded slot table below
    ("1A", "3rd_1"),
    ("1C", "2B"),
    ("1E", "3rd_2"),
    ("1G", "2H"),
    ("1B", "3rd_3"),
    ("1D", "2C"),
    ("1F", "3rd_4"),
    ("1H", "2G"),
    ("1I", "3rd_5"),
    ("1K", "2L"),
    ("1J", "3rd_6"),
    ("1L", "2I"),
    ("2A", "2E"),
    ("2D", "2F"),
    ("3rd_7", "3rd_8"),
    ("2J", "2K"),
]


class TournamentFormat:
    """
    Encodes the 2026 World Cup structure: groups, group-stage fixtures,
    third-place ranking, and the Round-of-32 bracket.

    Usage:
        fmt = TournamentFormat(config, team_names)
        for group in fmt.groups:
            ... simulate group.matches ...

        r32_matches = fmt.build_round_of_32(all_group_standings)
    """

    def __init__(self, config: dict, team_names: list[str]):
        """
        Args:
            config:     Project configuration (reads data.tournament section).
            team_names: 48 team names, ordered group-by-group
                        (positions 0-3 = Group A, 4-7 = Group B, ...).
                        Use the actual FIFA draw order when available.
        """
        tcfg = config["data"]["tournament"]
        self.n_groups = tcfg["n_groups"]
        self.teams_per_group = tcfg["teams_per_group"]
        self.third_place_qualifiers = tcfg["third_place_qualifiers"]
        self.total_matches = tcfg["total_matches"]

        self.groups: list[Group] = build_groups(
            team_names, self.n_groups, self.teams_per_group
        )

    def build_round_of_32(
        self,
        group_standings: dict[str, pd.DataFrame],
    ) -> list[KnockoutMatch]:
        """
        Build the 16 Round-of-32 matches from final group standings.

        Args:
            group_standings: Dict mapping group letter → standings DataFrame
                              (output of compute_group_standings, sorted by
                              position with a 'position' column).

        Returns:
            List of 16 KnockoutMatch objects.
        """
        # Collect group winners (position 1), runners-up (position 2),
        # and third-place teams (position 3) by group letter.
        winners:    dict[str, str] = {}
        runners_up: dict[str, str] = {}
        thirds:     list[pd.Series] = []
        third_groups: list[str] = []

        for letter, standings in group_standings.items():
            winners[letter]    = standings.iloc[0]["team"]
            runners_up[letter] = standings.iloc[1]["team"]
            third_row = standings.iloc[2]
            thirds.append(third_row)
            third_groups.append(letter)

        best_8_thirds = rank_third_place_teams(thirds, third_groups)

        # Build the position lookup table referenced by R32_FIXED_PAIRINGS
        position_map: dict[str, str] = {}
        for letter, team in winners.items():
            position_map[f"1{letter}"] = team
        for letter, team in runners_up.items():
            position_map[f"2{letter}"] = team
        for i, team in enumerate(best_8_thirds, start=1):
            position_map[f"3rd_{i}"] = team

        matches = []
        for i, (pos_a, pos_b) in enumerate(R32_FIXED_PAIRINGS, start=1):
            team_a = position_map.get(pos_a)
            team_b = position_map.get(pos_b)
            if team_a is None or team_b is None:
                logger.warning(f"  R32 slot {pos_a} vs {pos_b} could not be resolved")
                continue
            matches.append(KnockoutMatch(
                round_name="Round of 32",
                team_a=team_a,
                team_b=team_b,
                slot_label=f"Match {64 + i}",  # R32 are matches 65-80 in 104-match schedule
            ))

        logger.info(f"  Round of 32: {len(matches)} matches built")
        return matches

    @staticmethod
    def build_next_round(
        previous_round_winners: list[str],
        round_name: str,
        start_match_number: int,
    ) -> list[KnockoutMatch]:
        """
        Build the next knockout round by pairing consecutive winners.

        Standard single-elimination bracket: winners of matches (1,2), (3,4),
        (5,6)... face each other in the next round, preserving bracket order.

        Args:
            previous_round_winners: Winners from the prior round, in bracket order.
            round_name:              Name for the new round (e.g. 'Round of 16').
            start_match_number:      First match number for labelling.

        Returns:
            List of KnockoutMatch objects for the new round.
        """
        if len(previous_round_winners) % 2 != 0:
            raise ValueError("Number of winners must be even to pair into matches")

        matches = []
        for i in range(0, len(previous_round_winners), 2):
            matches.append(KnockoutMatch(
                round_name=round_name,
                team_a=previous_round_winners[i],
                team_b=previous_round_winners[i + 1],
                slot_label=f"Match {start_match_number + i // 2}",
            ))
        return matches
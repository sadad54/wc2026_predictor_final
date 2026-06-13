"""
tests/test_simulation.py

Unit tests for Phase 6 (match simulator) and Phase 7 (tournament simulator).

Run with: pytest tests/test_simulation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel
from src.simulation.match_simulator import MatchSimulator
from src.simulation.player_models import PlayerScoringModel
from src.simulation.tournament_format import (
    Group,
    TournamentFormat,
    build_groups,
    compute_group_standings,
    rank_third_place_teams,
)
from src.simulation.tournament_simulator import TournamentSimulator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dc_model():
    """Fitted Dixon-Coles model on synthetic data covering 48 teams."""
    rng = np.random.default_rng(0)
    teams = [f"Team{i:02d}" for i in range(48)]
    n = 600
    home = rng.choice(teams, n)
    away = rng.choice(teams, n)
    df = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=n, freq="3D"),
        "home_team": home,
        "away_team": away,
        "home_score": rng.integers(0, 5, n),
        "away_score": rng.integers(0, 4, n),
    })
    return DixonColesModel(xi=0.001).fit(df)


@pytest.fixture
def team_names():
    return [f"Team{i:02d}" for i in range(48)]


@pytest.fixture
def player_model(team_names):
    return PlayerScoringModel.placeholder(team_names)


@pytest.fixture
def minimal_config():
    return {
        "data": {
            "tournament": {
                "year": 2026,
                "n_teams": 48,
                "n_groups": 12,
                "teams_per_group": 4,
                "third_place_qualifiers": 8,
                "total_matches": 104,
            }
        },
        "simulation": {
            "n_simulations": 5,
            "random_state": 42,
            "knockout": {
                "extra_time_goal_rate": 0.75,
                "penalty_base_success": 0.753,
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MatchSimulator
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchSimulator:

    def test_group_match_can_be_a_draw(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        rng = np.random.default_rng(1)
        # Run several times — at least one should be a draw with a 48-team random model
        results = [
            sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=False, rng=rng)
            for _ in range(30)
        ]
        assert any(r.is_draw for r in results)

    def test_group_match_draw_has_no_winner(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        rng = np.random.default_rng(1)
        results = [
            sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=False, rng=rng)
            for _ in range(30)
        ]
        draws = [r for r in results if r.is_draw]
        assert all(r.winner is None for r in draws)

    def test_knockout_match_always_has_winner(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        rng = np.random.default_rng(2)
        for _ in range(30):
            result = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=True, rng=rng)
            assert result.winner is not None
            assert result.winner in ("Team00", "Team01")

    def test_knockout_extra_time_only_when_drawn(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        rng = np.random.default_rng(3)
        for _ in range(30):
            result = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=True, rng=rng)
            if not result.went_to_extra_time:
                assert result.home_goals != result.away_goals

    def test_penalties_produce_distinct_score(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        rng = np.random.default_rng(4)
        for _ in range(20):
            result = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=True, rng=rng)
            if result.went_to_penalties:
                assert result.pen_home_score != result.pen_away_score

    def test_scorers_match_goal_count(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        rng = np.random.default_rng(5)
        result = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=False, rng=rng)
        home_scorers = [p for t, p in result.scorers if t == "Team00"]
        away_scorers = [p for t, p in result.scorers if t == "Team01"]
        assert len(home_scorers) == result.home_goals
        assert len(away_scorers) == result.away_goals

    def test_reproducible_with_same_seed(self, dc_model, player_model):
        sim = MatchSimulator(dc_model, player_model)
        r1 = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=True,
                           rng=np.random.default_rng(99))
        r2 = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=True,
                           rng=np.random.default_rng(99))
        assert r1.home_goals == r2.home_goals
        assert r1.away_goals == r2.away_goals
        assert r1.winner == r2.winner

    def test_no_player_model_skips_scorers(self, dc_model):
        sim = MatchSimulator(dc_model, player_model=None)
        rng = np.random.default_rng(6)
        result = sim.simulate("Team00", "Team01", is_neutral=True, is_knockout=False, rng=rng)
        assert result.scorers == []
        assert result.yellow_cards == []


# ─────────────────────────────────────────────────────────────────────────────
# PlayerScoringModel
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerScoringModel:

    def test_placeholder_creates_11_players_per_team(self, team_names):
        model = PlayerScoringModel.placeholder(team_names)
        for team in team_names:
            assert len(model.team_players[team]) == 11

    def test_weights_sum_to_one(self, player_model, team_names):
        for team in team_names:
            assert np.isclose(player_model.team_weights[team].sum(), 1.0)

    def test_sample_scorer_returns_valid_player(self, player_model):
        rng = np.random.default_rng(0)
        player = player_model.sample_scorer("Team00", rng)
        assert player in player_model.team_players["Team00"]

    def test_unknown_team_does_not_crash(self, player_model):
        rng = np.random.default_rng(0)
        player = player_model.sample_scorer("AtlantisFC", rng)
        assert "Unknown" in player

    def test_from_squad_data(self):
        squads = pd.DataFrame({
            "team": ["A", "A", "B"],
            "player": ["A1", "A2", "B1"],
            "career_goals": [10, 0, 5],
            "career_appearances": [20, 1, 10],
        })
        model = PlayerScoringModel.from_squad_data(squads, min_appearances=5)
        assert set(model.team_players["A"]) == {"A1", "A2"}
        # A1 has higher goal rate than A2 (which falls back to 0.05)
        weights = dict(zip(model.team_players["A"], model.team_weights["A"]))
        assert weights["A1"] > weights["A2"]


# ─────────────────────────────────────────────────────────────────────────────
# Tournament format
# ─────────────────────────────────────────────────────────────────────────────

class TestTournamentFormat:

    def test_build_groups_count(self, team_names):
        groups = build_groups(team_names, n_groups=12, teams_per_group=4)
        assert len(groups) == 12
        assert all(len(g.teams) == 4 for g in groups)

    def test_build_groups_wrong_team_count_raises(self):
        with pytest.raises(ValueError):
            build_groups(["A", "B", "C"], n_groups=12, teams_per_group=4)

    def test_group_has_6_fixtures(self):
        group = Group(name="A", teams=["T1", "T2", "T3", "T4"])
        assert len(group.matches) == 6

    def test_group_standings_points_calculation(self):
        group = Group(name="A", teams=["T1", "T2", "T3", "T4"])
        # T1 beats everyone, T2 draws T3, T4 loses everything
        results = {
            ("T1", "T2"): (2, 0),
            ("T1", "T3"): (2, 0),
            ("T1", "T4"): (2, 0),
            ("T2", "T3"): (1, 1),
            ("T2", "T4"): (2, 0),
            ("T3", "T4"): (2, 0),
        }
        standings = compute_group_standings(group, results)
        assert standings.iloc[0]["team"] == "T1"
        assert standings.iloc[0]["points"] == 9
        assert standings.iloc[-1]["team"] == "T4"
        assert standings.iloc[-1]["points"] == 0

    def test_group_standings_has_position_column(self):
        group = Group(name="A", teams=["T1", "T2", "T3", "T4"])
        results = {pair: (1, 1) for pair in group.matches}
        standings = compute_group_standings(group, results)
        assert list(standings["position"]) == [1, 2, 3, 4]

    def test_rank_third_place_returns_8(self):
        rows = []
        for i in range(12):
            rows.append(pd.Series({
                "team": f"T{i}",
                "points": 12 - i,  # strictly decreasing → deterministic order
                "goal_difference": 0,
                "goals_for": 0,
            }))
        groups = [chr(ord("A") + i) for i in range(12)]
        best_8 = rank_third_place_teams(rows, groups)
        assert len(best_8) == 8
        # Highest points should be first
        assert best_8[0] == "T0"

    def test_rank_third_place_wrong_count_raises(self):
        with pytest.raises(ValueError):
            rank_third_place_teams([pd.Series({"team": "T1", "points": 1,
                                                  "goal_difference": 0, "goals_for": 0})],
                                    ["A"])

    def test_round_of_32_produces_16_matches(self, team_names, minimal_config):
        fmt = TournamentFormat(minimal_config, team_names)
        # Build trivial standings: alphabetical order = positions 1-4 in each group
        group_standings = {}
        for group in fmt.groups:
            df = pd.DataFrame({
                "team": group.teams,
                "points": [9, 6, 3, 0],
                "goal_difference": [5, 2, -1, -6],
                "goals_for": [10, 6, 3, 1],
            })
            df["position"] = range(1, 5)
            group_standings[group.name] = df

        r32 = fmt.build_round_of_32(group_standings)
        assert len(r32) == 16

    def test_build_next_round_pairs_correctly(self):
        winners = ["A", "B", "C", "D"]
        matches = TournamentFormat.build_next_round(winners, "Quarterfinal", 89)
        assert len(matches) == 2
        assert (matches[0].team_a, matches[0].team_b) == ("A", "B")
        assert (matches[1].team_a, matches[1].team_b) == ("C", "D")

    def test_build_next_round_odd_count_raises(self):
        with pytest.raises(ValueError):
            TournamentFormat.build_next_round(["A", "B", "C"], "X", 1)


# ─────────────────────────────────────────────────────────────────────────────
# TournamentSimulator
# ─────────────────────────────────────────────────────────────────────────────

class TestTournamentSimulator:

    def test_simulate_one_tournament_has_champion(self, minimal_config, dc_model, team_names, player_model):
        sim = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        rng = np.random.default_rng(0)
        result = sim.simulate_one_tournament(rng)
        assert result["champion"] in team_names
        assert result["runner_up"] in team_names
        assert result["champion"] != result["runner_up"]

    def test_champion_reaches_champion_stage(self, minimal_config, dc_model, team_names, player_model):
        sim = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        rng = np.random.default_rng(0)
        result = sim.simulate_one_tournament(rng)
        assert result["stage_reached"][result["champion"]] == "champion"

    def test_all_48_teams_have_a_stage(self, minimal_config, dc_model, team_names, player_model):
        sim = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        rng = np.random.default_rng(0)
        result = sim.simulate_one_tournament(rng)
        assert set(result["stage_reached"].keys()) == set(team_names)

    def test_run_monte_carlo_probabilities_sum_to_one(self, minimal_config, dc_model, team_names, player_model):
        sim = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        results = sim.run_monte_carlo(n_simulations=10)
        total = results["final_positions"]["p_champion"].sum()
        assert abs(total - 1.0) < 1e-6

    def test_run_monte_carlo_reproducible(self, minimal_config, dc_model, team_names, player_model):
        sim1 = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        sim2 = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        r1 = sim1.run_monte_carlo(n_simulations=5)
        r2 = sim2.run_monte_carlo(n_simulations=5)
        pd.testing.assert_frame_equal(
            r1["final_positions"].reset_index(drop=True),
            r2["final_positions"].reset_index(drop=True),
        )

    def test_golden_boot_total_goals_positive(self, minimal_config, dc_model, team_names, player_model):
        sim = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        results = sim.run_monte_carlo(n_simulations=5)
        assert (results["golden_boot"]["mean_goals"] >= 0).all()

    def test_team_progress_monotonically_decreasing(self, minimal_config, dc_model, team_names, player_model):
        """A team can't reach the final without reaching the semifinal, etc."""
        sim = TournamentSimulator(minimal_config, dc_model, team_names, player_model)
        results = sim.run_monte_carlo(n_simulations=20)
        progress = results["team_progress"]
        stages = ["reach_group_stage", "reach_round_of_32", "reach_round_of_16",
                  "reach_quarterfinal", "reach_semifinal", "reach_champion"]
        for _, row in progress.iterrows():
            values = [row[s] for s in stages]
            assert all(values[i] >= values[i+1] for i in range(len(values) - 1))
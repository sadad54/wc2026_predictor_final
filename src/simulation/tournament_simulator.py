"""
src/simulation/tournament_simulator.py

Phase 7: Full tournament Monte Carlo simulation.

One call to simulate_one_tournament() plays out all 104 matches:
    Group stage (72 matches) → standings → Round of 32 (16) →
    Round of 16 (8) → Quarterfinals (4) → Semifinals (2) →
    Third-place playoff (1) + Final (1)

run_monte_carlo() repeats this n_simulations times (default 10,000),
collecting per-team and per-player outcomes across all runs.

Each run uses an independently-seeded child Generator derived from a
master seed, so the WHOLE simulation is reproducible: re-running with
the same config.simulation.random_state reproduces every individual
tournament exactly.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.models.dixon_coles import DixonColesModel
from src.simulation.match_simulator import MatchSimulator, MatchResult
from src.simulation.player_models import PlayerScoringModel
from src.simulation.tournament_format import (
    TournamentFormat,
    compute_group_standings,
)


class TournamentSimulator:
    """
    Runs Monte Carlo simulations of the full 2026 World Cup.

    Usage:
        sim = TournamentSimulator(config, dc_model, team_names, player_model)
        results = sim.run_monte_carlo(n_simulations=10000)

        results['team_progress']     — DataFrame: team x stage probabilities
        results['final_positions']   — DataFrame: P(champion), P(runner-up), etc.
        results['golden_boot']       — DataFrame: player x P(top scorer)
    """

    def __init__(
        self,
        config: dict,
        dc_model: DixonColesModel,
        team_names: list[str],
        player_model: PlayerScoringModel | None = None,
    ):
        """
        Args:
            config:       Project configuration.
            dc_model:     Fitted DixonColesModel (provides match probabilities).
            team_names:   48 team names, ordered group-by-group (see
                          TournamentFormat for ordering requirements).
            player_model: PlayerScoringModel for goal-scorer tracking.
                          If None, a placeholder model is built automatically.
        """
        self.config = config
        self.dc_model = dc_model
        self.team_names = team_names

        if player_model is None:
            player_model = PlayerScoringModel.placeholder(team_names)
        self.player_model = player_model

        et_rate = config["simulation"]["knockout"]["extra_time_goal_rate"]
        pen_rate = config["simulation"]["knockout"]["penalty_base_success"]
        self.match_sim = MatchSimulator(
            dc_model=dc_model,
            player_model=player_model,
            et_goal_rate=et_rate,
            pen_base_success=pen_rate,
        )

        self.format = TournamentFormat(config, team_names)

    # ── Single tournament run ─────────────────────────────────────────────────

    def simulate_one_tournament(self, rng: np.random.Generator) -> dict:
        """
        Simulate one complete 104-match tournament.

        Args:
            rng: Seeded NumPy Generator for this run.

        Returns:
            Dict with:
                'stage_reached':   {team: stage_name} — furthest stage each
                                    team reached ('group_stage', 'round_of_32',
                                    ..., 'champion')
                'champion':        team name
                'runner_up':       team name
                'third_place':     team name
                'top_scorers':     Counter-like dict {player: n_goals} for
                                    THIS run only
        """
        stage_reached: dict[str, str] = {t: "group_stage" for t in self.team_names}
        goals_this_run: dict[str, int] = defaultdict(int)

        # ── Group stage ──────────────────────────────────────────────────────
        group_standings: dict[str, pd.DataFrame] = {}

        for group in self.format.groups:
            match_results: dict[tuple[str, str], tuple[int, int]] = {}

            for home, away in group.matches:
                result = self.match_sim.simulate(
                    home_team=home,
                    away_team=away,
                    is_neutral=True,
                    is_knockout=False,
                    rng=rng,
                )
                match_results[(home, away)] = (result.home_goals, result.away_goals)
                self._tally_goals(result, goals_this_run)

            standings = compute_group_standings(group, match_results)
            group_standings[group.name] = standings

            # Top 2 advance to "round_of_32" stage marker
            for pos, team in enumerate(standings["team"].tolist(), start=1):
                if pos <= 2:
                    stage_reached[team] = "round_of_32"

        # ── Determine third-place qualifiers ────────────────────────────────
        r32_matches = self.format.build_round_of_32(group_standings)

        # Any team appearing in r32_matches advances (covers best-8-thirds too)
        for m in r32_matches:
            stage_reached[m.team_a] = "round_of_32"
            stage_reached[m.team_b] = "round_of_32"

        # ── Round of 32 ───────────────────────────────────────────────────────
        r16_qualifiers = self._play_knockout_round(
            r32_matches, "round_of_16", stage_reached, goals_this_run, rng
        )

        # ── Round of 16 ───────────────────────────────────────────────────────
        r16_matches = TournamentFormat.build_next_round(
            r16_qualifiers, "Round of 16", start_match_number=81
        )
        qf_qualifiers = self._play_knockout_round(
            r16_matches, "quarterfinal", stage_reached, goals_this_run, rng
        )

        # ── Quarterfinals ─────────────────────────────────────────────────────
        qf_matches = TournamentFormat.build_next_round(
            qf_qualifiers, "Quarterfinal", start_match_number=89
        )
        sf_qualifiers = self._play_knockout_round(
            qf_matches, "semifinal", stage_reached, goals_this_run, rng
        )

        # ── Semifinals ────────────────────────────────────────────────────────
        sf_matches = TournamentFormat.build_next_round(
            sf_qualifiers, "Semifinal", start_match_number=93
        )
        finalists = []
        sf_losers = []
        for m in sf_matches:
            result = self.match_sim.simulate(
                home_team=m.team_a, away_team=m.team_b,
                is_neutral=True, is_knockout=True, rng=rng,
            )
            self._tally_goals(result, goals_this_run)
            winner = result.winner
            loser = m.team_b if winner == m.team_a else m.team_a
            finalists.append(winner)
            sf_losers.append(loser)
            stage_reached[winner] = "final"
            stage_reached[loser] = "semifinal"

        # ── Third-place playoff ──────────────────────────────────────────────
        third_place_result = self.match_sim.simulate(
            home_team=sf_losers[0], away_team=sf_losers[1],
            is_neutral=True, is_knockout=True, rng=rng,
        )
        self._tally_goals(third_place_result, goals_this_run)
        third_place_team = third_place_result.winner

        # ── Final ─────────────────────────────────────────────────────────────
        final_result = self.match_sim.simulate(
            home_team=finalists[0], away_team=finalists[1],
            is_neutral=True, is_knockout=True, rng=rng,
        )
        self._tally_goals(final_result, goals_this_run)
        champion = final_result.winner
        runner_up = finalists[1] if champion == finalists[0] else finalists[0]

        stage_reached[champion] = "champion"
        stage_reached[runner_up] = "runner_up"
        stage_reached[third_place_team] = "third_place"
        # The other third-place-playoff loser stays at "semifinal"

        return {
            "stage_reached": stage_reached,
            "champion": champion,
            "runner_up": runner_up,
            "third_place": third_place_team,
            "top_scorers": dict(goals_this_run),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _play_knockout_round(
        self,
        matches: list,
        next_stage_label: str,
        stage_reached: dict[str, str],
        goals_this_run: dict[str, int],
        rng: np.random.Generator,
    ) -> list[str]:
        """
        Simulate every match in a knockout round.

        Updates stage_reached: winners advance to next_stage_label,
        losers stay at their current stage (their max reached so far).

        Returns:
            List of winners, in bracket order — feeds into the next round.
        """
        winners = []
        for m in matches:
            result = self.match_sim.simulate(
                home_team=m.team_a, away_team=m.team_b,
                is_neutral=True, is_knockout=True, rng=rng,
            )
            self._tally_goals(result, goals_this_run)
            winner = result.winner
            winners.append(winner)
            stage_reached[winner] = next_stage_label
        return winners

    @staticmethod
    def _tally_goals(result: MatchResult, goals_this_run: dict[str, int]) -> None:
        """Increment per-player goal counts for one match's scorers."""
        for _team, player in result.scorers:
            goals_this_run[player] += 1

    # ── Monte Carlo orchestration ────────────────────────────────────────────

    def run_monte_carlo(self, n_simulations: int | None = None) -> dict:
        """
        Run n_simulations full tournament simulations and aggregate results.

        Args:
            n_simulations: Number of runs. Defaults to config['simulation']['n_simulations'].

        Returns:
            Dict with three DataFrames:
                'final_positions': team x {P(champion), P(runner_up), P(third_place),
                                            P(semifinal), P(quarterfinal), P(round_of_16),
                                            P(round_of_32), P(group_stage_exit)}
                'team_progress':   team x cumulative reach probability per stage
                'golden_boot':     player x {team, P(top_scorer), mean_goals,
                                              P(at_least_5_goals)}
        """
        if n_simulations is None:
            n_simulations = self.config["simulation"]["n_simulations"]

        master_seed = self.config["simulation"]["random_state"]
        master_rng = np.random.default_rng(master_seed)

        logger.info(f"Running {n_simulations:,} tournament simulations...")
        logger.info(f"  Teams: {len(self.team_names)} | Groups: {len(self.format.groups)}")

        # Per-run results
        stage_counts: dict[str, dict[str, int]] = {
            t: defaultdict(int) for t in self.team_names
        }
        champion_counts: dict[str, int] = defaultdict(int)
        runner_up_counts: dict[str, int] = defaultdict(int)
        third_place_counts: dict[str, int] = defaultdict(int)

        # Golden Boot tracking
        top_scorer_wins: dict[str, int] = defaultdict(int)
        total_goals: dict[str, int] = defaultdict(int)
        at_least_5_goals: dict[str, int] = defaultdict(int)
        player_team: dict[str, str] = {}

        for i in tqdm(range(n_simulations), desc="Simulating tournaments"):
            # Independent child seed per run — full reproducibility
            run_rng = np.random.default_rng(master_seed + i)

            run_result = self.simulate_one_tournament(run_rng)

            for team, stage in run_result["stage_reached"].items():
                stage_counts[team][stage] += 1

            champion_counts[run_result["champion"]] += 1
            runner_up_counts[run_result["runner_up"]] += 1
            third_place_counts[run_result["third_place"]] += 1

            # Golden Boot for this run
            top_scorers = run_result["top_scorers"]
            for player, goals in top_scorers.items():
                total_goals[player] += goals
                if goals >= 5:
                    at_least_5_goals[player] += 1
                # Record player's team if not already known
                if player not in player_team:
                    player_team[player] = self._infer_team_from_player_name(player)

            if top_scorers:
                top_player = max(top_scorers.items(), key=lambda kv: kv[1])[0]
                top_scorer_wins[top_player] += 1

        # ── Build final_positions DataFrame ──────────────────────────────────
        stage_order = [
            "champion", "runner_up", "third_place", "semifinal",
            "quarterfinal", "round_of_16", "round_of_32", "group_stage",
        ]

        rows = []
        for team in self.team_names:
            counts = stage_counts[team]
            row = {"team": team}
            for stage in stage_order:
                row[f"p_{stage}"] = counts.get(stage, 0) / n_simulations
            rows.append(row)

        final_positions = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)

        # ── Build team_progress (cumulative "reached at least this stage") ───
        cumulative_order = [
            "group_stage", "round_of_32", "round_of_16",
            "quarterfinal", "semifinal", "champion",
        ]
        progress_rows = []
        for team in self.team_names:
            counts = stage_counts[team]
            row = {"team": team}
            # Map stage_reached labels to cumulative thresholds
            cum_at_least = {
                "group_stage": n_simulations,  # everyone plays the group stage
                "round_of_32": counts.get("round_of_32", 0) + counts.get("round_of_16", 0)
                               + counts.get("quarterfinal", 0) + counts.get("semifinal", 0)
                               + counts.get("champion", 0) + counts.get("runner_up", 0)
                               + counts.get("third_place", 0),
                "round_of_16": counts.get("round_of_16", 0) + counts.get("quarterfinal", 0)
                               + counts.get("semifinal", 0) + counts.get("champion", 0)
                               + counts.get("runner_up", 0) + counts.get("third_place", 0),
                "quarterfinal": counts.get("quarterfinal", 0) + counts.get("semifinal", 0)
                                + counts.get("champion", 0) + counts.get("runner_up", 0)
                                + counts.get("third_place", 0),
                "semifinal": counts.get("semifinal", 0) + counts.get("champion", 0)
                             + counts.get("runner_up", 0) + counts.get("third_place", 0),
                "champion": counts.get("champion", 0),
            }
            for stage in cumulative_order:
                row[f"reach_{stage}"] = cum_at_least[stage] / n_simulations
            progress_rows.append(row)

        team_progress = pd.DataFrame(progress_rows).sort_values(
            "reach_champion", ascending=False
        ).reset_index(drop=True)

        # ── Build golden_boot DataFrame ───────────────────────────────────────
        gb_rows = []
        for player, goals in total_goals.items():
            gb_rows.append({
                "player": player,
                "team": player_team.get(player, "Unknown"),
                "p_top_scorer": top_scorer_wins.get(player, 0) / n_simulations,
                "mean_goals": goals / n_simulations,
                "p_at_least_5_goals": at_least_5_goals.get(player, 0) / n_simulations,
            })

        golden_boot = pd.DataFrame(gb_rows).sort_values(
            "p_top_scorer", ascending=False
        ).reset_index(drop=True)

        logger.info("Monte Carlo simulation complete")
        logger.info(f"  Top 3 champions: {final_positions.head(3)[['team', 'p_champion']].to_dict('records')}")
        if not golden_boot.empty:
            logger.info(f"  Top Golden Boot: {golden_boot.iloc[0][['player', 'team', 'p_top_scorer']].to_dict()}")

        return {
            "final_positions": final_positions,
            "team_progress": team_progress,
            "golden_boot": golden_boot,
        }

    @staticmethod
    def _infer_team_from_player_name(player: str) -> str:
        """
        Best-effort extraction of team name from a placeholder player name
        of the form "{team} Player {n}". Returns 'Unknown' if it doesn't match.

        Once real squad data is used (player_model.from_squad_data), player
        names won't follow this pattern — replace this with a lookup against
        the squad DataFrame instead.
        """
        if " Player " in player:
            return player.split(" Player ")[0]
        return "Unknown"
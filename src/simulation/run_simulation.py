"""
src/simulation/run_simulation.py

Phase 7 entry point — full tournament Monte Carlo simulation.

Requires:
    - A fitted DixonColesModel (from Phase 4's ensemble, models/saved/dixon_coles.pkl)
    - List of 48 qualified teams for WC 2026, ordered group-by-group
      (this comes from the official FIFA draw — placeholder list used
      here until the real draw data is wired in)

Run:
    python -m src.simulation.run_simulation
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.models.dixon_coles import DixonColesModel
from src.simulation.player_models import PlayerScoringModel
from src.simulation.results_aggregator import (
    find_dark_horses,
    print_summary,
    save_simulation_results,
)
from src.simulation.tournament_simulator import TournamentSimulator
from src.utils.helpers import initialize_project


def load_team_names(config: dict) -> list[str]:
    """
    Load the 48 qualified teams, ordered group-by-group (A through L).

    Placeholder: until the real FIFA 2026 draw data is added to
    data/external/, this raises with instructions. Replace the body
    of this function with a load from data/external/wc2026_groups.csv
    once that file exists (columns: group, team).
    """
    groups_path = Path(config["paths"]["external_data"]) / "wc2026_groups.csv"

    if not groups_path.exists():
        raise FileNotFoundError(
            f"{groups_path} not found.\n"
            "Create this file with columns 'group' and 'team' listing all 48 "
            "qualified teams in group order (Group A teams first, then B, ... L).\n"
            "The official draw will be available closer to the tournament — "
            "for now you can use a placeholder seeding based on current FIFA rankings."
        )

    df = pd.read_csv(groups_path)
    df = df.sort_values("group")
    return df["team"].tolist()


def run_simulation(config: dict) -> dict:
    models_dir = Path(config["paths"]["models"])
    processed_dir = Path(config["paths"]["processed_data"])
    metrics_dir = Path(config["paths"]["metrics"]) / "simulation"

    # ── Load fitted Dixon-Coles model ────────────────────────────────────────
    dc_path = models_dir / "dixon_coles.pkl"
    if not dc_path.exists():
        raise FileNotFoundError(f"{dc_path} not found — run Phase 4 (train.py) first")

    dc_model = DixonColesModel.load(dc_path)

    # ── Load qualified teams ──────────────────────────────────────────────────
    team_names = load_team_names(config)
    logger.info(f"Loaded {len(team_names)} teams for simulation")

    # Warn about any teams missing from the Dixon-Coles model
    missing = [t for t in team_names if not dc_model.has_team(t)]
    if missing:
        logger.warning(
            f"{len(missing)} teams not in Dixon-Coles training data "
            f"(will use fallback probabilities): {missing}"
        )

    # ── Player model (placeholder until squad data is collected) ────────────
    player_model = PlayerScoringModel.placeholder(team_names)

    # ── Run simulation ─────────────────────────────────────────────────────────
    simulator = TournamentSimulator(config, dc_model, team_names, player_model)
    results = simulator.run_monte_carlo()

    # ── Save + summarise ──────────────────────────────────────────────────────
    save_simulation_results(results, metrics_dir)
    print_summary(results)

    # Dark horse analysis
    rankings_path = processed_dir / "rankings_clean.parquet"
    if rankings_path.exists():
        rankings_df = pd.read_parquet(rankings_path)
        find_dark_horses(results["final_positions"], rankings_df)

    logger.info(f"\n✅ Phase 7 complete — results in {metrics_dir}/")
    return results


if __name__ == "__main__":
    config = initialize_project()
    run_simulation(config)
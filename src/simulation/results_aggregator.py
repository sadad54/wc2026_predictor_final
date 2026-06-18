"""
src/simulation/results_aggregator.py

Phase 7: Save and summarise Monte Carlo simulation results.

Saves three CSVs to models/metrics/simulation/:
    final_positions.csv  — P(champion), P(runner-up), etc. per team
    team_progress.csv    — cumulative stage-reach probabilities
    golden_boot.csv      — player-level Golden Boot probabilities

Also prints a "dark horse" report: teams whose simulated title probability
significantly exceeds their FIFA ranking position would suggest — exactly
the framing called for in the project plan.

Run as part of Phase 7's main script (see run_simulation.py below).
"""

from pathlib import Path

import pandas as pd
from loguru import logger


def save_simulation_results(results: dict, output_dir: Path) -> None:
    """
    Persist all three simulation result DataFrames to CSV.

    Args:
        results:    Output of TournamentSimulator.run_monte_carlo().
        output_dir: Directory to save into (created if missing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results["final_positions"].to_csv(output_dir / "final_positions.csv", index=False)
    results["team_progress"].to_csv(output_dir / "team_progress.csv", index=False)
    results["golden_boot"].to_csv(output_dir / "golden_boot.csv", index=False)
    if "simulation_diagnostics" in results:
        results["simulation_diagnostics"].to_csv(
            output_dir / "simulation_diagnostics.csv", index=False
        )

    logger.info(f"Saved simulation results → {output_dir}/")


def print_summary(results: dict, top_n: int = 10) -> None:
    """
    Print a readable summary of the simulation: title favourites,
    Golden Boot favourites, and basic sanity checks.

    Args:
        results: Output of run_monte_carlo().
        top_n:   Number of teams/players to show in each list.
    """
    fp = results["final_positions"]
    gb = results["golden_boot"]

    logger.info("\n" + "=" * 60)
    logger.info("TOURNAMENT SIMULATION SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nTop {top_n} title favourites:")
    for _, row in fp.head(top_n).iterrows():
        logger.info(f"  {row['team']:<20} {row['p_champion']*100:5.1f}% champion | "
                     f"{row['p_runner_up']*100:5.1f}% runner-up | "
                     f"{row['p_semifinal']*100:5.1f}% reach SF")

    if not gb.empty:
        logger.info(f"\nTop {top_n} Golden Boot favourites:")
        for _, row in gb.head(top_n).iterrows():
            logger.info(f"  {row['player']:<25} ({row['team']:<15}) "
                         f"{row['p_top_scorer']*100:5.1f}% | "
                         f"avg {row['mean_goals']:.2f} goals")

    # Sanity check: probabilities should sum sensibly
    total_champion_prob = fp["p_champion"].sum()
    logger.info(f"\nSanity check — sum of P(champion) across all teams: {total_champion_prob:.4f} (should be ~1.0)")


def find_dark_horses(
    final_positions: pd.DataFrame,
    rankings_df: pd.DataFrame,
    ranking_date_col: str = "rank_date",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify "dark horse" teams: model title-probability rank is
    significantly better than their current FIFA ranking position.

    This is the rigorous version of "dark horse" called for in the
    project plan — a team whose simulated probability of winning
    outranks where their FIFA ranking alone would place them.

    Args:
        final_positions: Output of run_monte_carlo()['final_positions'].
        rankings_df:      Cleaned rankings DataFrame (rankings_clean.parquet).
        ranking_date_col: Column with ranking date — uses the most recent entry.
        top_n:            Number of dark horses to return.

    Returns:
        DataFrame with columns: team, model_rank, fifa_rank,
        rank_difference (positive = model is much more bullish than FIFA),
        p_champion.
    """
    # Most recent FIFA rank per team
    latest_rankings = (
        rankings_df.sort_values(ranking_date_col)
        .groupby("team")
        .tail(1)[["team", "rank"]]
        .rename(columns={"rank": "fifa_rank"})
    )

    df = final_positions.copy()
    df["model_rank"] = df["p_champion"].rank(ascending=False, method="min").astype(int)

    df = df.merge(latest_rankings, on="team", how="left")
    df["fifa_rank"] = df["fifa_rank"].fillna(df["fifa_rank"].max())

    # Positive = model thinks this team is much better than FIFA ranking suggests
    df["rank_difference"] = df["fifa_rank"] - df["model_rank"]

    dark_horses = (
        df.sort_values("rank_difference", ascending=False)
        .head(top_n)[["team", "model_rank", "fifa_rank", "rank_difference", "p_champion"]]
        .reset_index(drop=True)
    )

    logger.info(f"\nDark horses (model rank much better than FIFA rank):")
    for _, row in dark_horses.iterrows():
        logger.info(
            f"  {row['team']:<20} model rank #{int(row['model_rank']):<3} "
            f"vs FIFA rank #{int(row['fifa_rank']):<3} "
            f"(P(champion)={row['p_champion']*100:.2f}%)"
        )

    return dark_horses

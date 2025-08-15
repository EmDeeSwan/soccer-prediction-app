import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class MLSNPRegSeasonPredictor:
    """
    A pure computational engine for running MLS Next Pro regular season simulations.
    
    This class takes pre-fetched game data and team statistics to run Monte Carlo
    simulations. It does not perform any I/O operations (database or API calls),
    which is handled in the database manager.
    """

    def __init__(self, conference: str, conference_teams: Dict[str, str], games_data: List[Dict], team_performance: Dict[str, Dict], league_averages: Dict[str, float]):
        """
        Initialize the predictor with all necessary data.
        
        Args:
            conference (str): The conference to simulate ('eastern' or 'western').
            conference_teams (Dict[str, str]): A dictionary of team_id -> team_name for the conference.
            games_data (List[Dict]): A list of all game data for the season.
            team_performance (Dict[str, Dict]): Pre-calculated team performance metrics (xG, goals per game).
            league_averages (Dict[str, float]): League-wide average goals and xG.
        """
        logger.info(f"Initializing MLSNPRegSeasonPredictor for {conference} conference.")
        self.conference = conference
        self.conference_teams = set(conference_teams.keys())
        self.team_names = conference_teams
        self.games_data = games_data
        self.team_performance = team_performance
        self.league_avg_xgf = league_averages.get('league_avg_xgf', 1.2)
        self.league_avg_xga = league_averages.get('league_avg_xga', 1.2)
        logger.info(f"Using league averages for {conference} predictor: xGF={self.league_avg_xgf:.2f}, xGA={self.league_avg_xga:.2f}")

        # Constants for simulation
        self.HOME_SHOOTOUT_WIN_PROB = 0.55 # Assuming 55% chance for home team in shootout
        logger.info(f"Home shootout win probability set for {conference} predictor to: {self.HOME_SHOOTOUT_WIN_PROB}")

        self.current_standings = self._calculate_current_standings()
        self.remaining_games = self._filter_remaining_games()

        logger.info(f"=== Post-Initialization Data Overview for {conference} ===")
        logger.info(f"Total games passed to predictor: {len(self.games_data)}")
        logger.info(f"Conference teams: {len(self.conference_teams)}")
        logger.info(f"Remaining games after filtering: {len(self.remaining_games)}")
        
        # Let's see what's in the first few games for good measure
        for i, game in enumerate(self.games_data[:3]):
            logger.info(f"Sample game {i+1}: completed={game.get('is_completed')}, home={game.get('home_team_id')}, away={game.get('away_team_id')}")

    def _check_data_quality(self) -> List[str]:
        """
        Check for suspicious data patterns that might indicate incorrect game data.
        Returns list of warning messages.
        """
        warnings = []
        
        # Check for teams with unusual win/loss patterns
        for team_id, stats in self.current_standings.items():
            team_name = self.team_names.get(team_id, team_id)
            
            # Check for impossible records (more wins than games played, etc.)
            total_results = stats.get('wins', 0) + stats.get('losses', 0) + stats.get('draws', 0)
            games_played = stats.get('games_played', 0)

            if total_results != games_played and games_played > 0:
                warnings.append(f"⚠️  {team_name}: Win/loss/draw counts don't add up to games played ({total_results} vs {games_played})")
            
            if stats.get('wins', 0) > games_played:
                warnings.append(f"⚠️  {team_name}: More wins ({stats.get('wins', 0)}) than games played ({games_played})")
        
            if stats.get('losses', 0) > games_played:
                warnings.append(f"⚠️  {team_name}: More losses ({stats.get('losses', 0)}) than games played ({games_played})")
                
            if stats.get('draws', 0) > games_played:
                warnings.append(f"⚠️  {team_name}: More draws ({stats.get('draws', 0)}) than games played ({games_played})")
            
            # Check for unusual points calculations
            shootout_losses = stats.get('draws', 0) - stats.get('shootout_wins', 0)
            expected_points = (stats.get('wins', 0) * 3) + (stats.get('shootout_wins', 0) * 2) + (stats.get('draws', 0) - stats.get('shootout_wins', 0))
            actual_points = stats.get('points', 0)
            
            if abs(expected_points - actual_points) > 0:  # Allow for small rounding errors
                warnings.append(f"⚠️  {team_name}: Points calculation seems off (expected {expected_points}, got {actual_points})")
                warnings.append(f"    → Breakdown: {stats.get('wins', 0)} reg wins × 3 + {stats.get('shootout_wins', 0)} SO wins × 2 + {shootout_losses} SO losses × 1")
        
        # Check for corrected games
        corrected_games = [g for g in self.games_data if g.get('data_corrected', False)]
        if corrected_games:
            warnings.append(f"ℹ️  {len(corrected_games)} games were auto-corrected during import (check correction_notes)")
        
        return warnings
    
    def _debug_team_standings(self, team_id: str) -> None:
        """Debug helper to trace a specific team's standings calculation"""
        if team_id not in self.conference_teams:
            return
            
        logger.info(f"\n=== DEBUG: Tracing standings for {self.team_names.get(team_id, team_id)} ===")
        
        # Count different game types for this team
        reg_wins = reg_losses = shootout_games = 0
        
        for game in self.games_data:
            if not game.get("is_completed"):
                continue
                
            home_id, away_id = game["home_team_id"], game["away_team_id"]
            if team_id not in [home_id, away_id]:
                continue
                
            try:
                home_score = int(game.get("home_score", 0))
                away_score = int(game.get("away_score", 0))
            except (ValueError, TypeError):
                continue
                
            went_to_shootout = game.get("went_to_shootout", False)
            
            if team_id == home_id:
                team_score, opp_score = home_score, away_score
            else:
                team_score, opp_score = away_score, home_score
                
            if went_to_shootout:
                shootout_games += 1
                logger.info(f"  Shootout game: {team_score}-{opp_score} (reg), SO result: {game.get('home_penalties', 0)}-{game.get('away_penalties', 0)}")
            elif team_score > opp_score:
                reg_wins += 1
                logger.info(f"  Regular win: {team_score}-{opp_score}")
            elif team_score < opp_score:
                reg_losses += 1
                logger.info(f"  Regular loss: {team_score}-{opp_score}")
        
        final_stats = self.current_standings.get(team_id, {})
        logger.info(f"Final calculated stats:")
        logger.info(f"  Games played: {final_stats.get('games_played', 0)}")
        logger.info(f"  Wins: {final_stats.get('wins', 0)} (expected reg wins: {reg_wins})")
        logger.info(f"  Losses: {final_stats.get('losses', 0)} (expected reg losses: {reg_losses})")
        logger.info(f"  Draws: {final_stats.get('draws', 0)} (expected shootout games: {shootout_games})")
        logger.info(f"  Shootout wins: {final_stats.get('shootout_wins', 0)}")
        logger.info(f"  Points: {final_stats.get('points', 0)}")
        logger.info("=== END DEBUG ===\n")
    
    def _calculate_current_standings(self) -> Dict[str, Dict]:
        """
        Calculates current standings based on completed games from the provided data.
        This mirrors the logic from DatabaseManager but operates on the local data copy.
        """
        standings = defaultdict(lambda: {
            "team_id": None, "name": "", "points": 0, "goal_difference": 0,
            "games_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "shootout_wins": 0
        })

        completed_games_count = 0
        shootout_games_count = 0
        skipped_games_count = 0

        for team_id in self.conference_teams:
            standings[team_id]["team_id"] = team_id
            standings[team_id]["name"] = self.team_names.get(team_id, f"Team {team_id}")

        for game in self.games_data:
            is_completed = game.get("is_completed", False)

            if not game.get("is_completed", False):
                skipped_games_count += 1
                continue

            home_id, away_id = game["home_team_id"], game["away_team_id"]
            
            # Ensure both teams are in our conference
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue

            # Score handling with validation
            try:
                home_score = int(game.get("home_score", 0))
                away_score = int(game.get("away_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid scores for game {game.get('game_id', 'unknown')}: {game.get('home_score')} - {game.get('away_score')}")
                continue

            completed_games_count += 1

            # Initialize team data if needed
            for team_id in [home_id, away_id]:
                if standings[team_id]["team_id"] is None:
                    standings[team_id]["team_id"] = team_id
                    standings[team_id]["name"] = self.team_names.get(team_id, f"Team {team_id}")

            # Shootout handling
            went_to_shootout = game.get("went_to_shootout", False)
            
            if went_to_shootout:
                shootout_games_count += 1
                home_pens = game.get("home_penalties", 0) or 0
                away_pens = game.get("away_penalties", 0) or 0
                
                # STEP 1: Both teams get regulation draw stats
                # (games_played, goals_for/against, draws)
                self._update_regulation_draw(standings[home_id], home_score, away_score)
                self._update_regulation_draw(standings[away_id], away_score, home_score)
                
                # STEP 2: Award shootout result
                if home_pens > away_pens:  # Home wins shootout
                    standings[home_id]["shootout_wins"] += 1
                    standings[home_id]["points"] += 2  # 2 total points for SO win
                    standings[away_id]["points"] += 1   # 1 point for SO loss
                    logger.debug(f"Shootout: {home_id} beats {away_id} {home_pens}-{away_pens}")
                else:  # Away wins shootout
                    standings[away_id]["shootout_wins"] += 1
                    standings[away_id]["points"] += 2  # 2 total points for SO win
                    standings[home_id]["points"] += 1   # 1 point for SO loss
                    logger.debug(f"Shootout: {away_id} beats {home_id} {away_pens}-{home_pens}")
                    
            else:
                # Regular time result (no shootout)
                if home_score > away_score:
                    self._update_team_standings(standings[home_id], home_score, away_score, "win")
                    self._update_team_standings(standings[away_id], away_score, home_score, "loss")
                elif away_score > home_score:
                    self._update_team_standings(standings[away_id], away_score, home_score, "win")
                    self._update_team_standings(standings[home_id], home_score, away_score, "loss")
                else:
                    # This shouldn't happen in MLS Next Pro (all draws go to shootout)
                    logger.warning(f"Regulation draw without shootout in game {game.get('game_id', 'unknown')}")
                    self._update_regulation_draw(standings[home_id], home_score, away_score)
                    self._update_regulation_draw(standings[away_id], away_score, home_score)
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1

        # Calculate final goal differences
        for team_id, stats in standings.items():
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]

        logger.info(f"Processed {completed_games_count} completed games ({shootout_games_count} went to shootout)")
        
        return {team_id: dict(stats) for team_id, stats in standings.items()}

    def _filter_remaining_games(self) -> List[Dict]:
        """Filters for future games to be simulated."""
        return [
            game for game in self.games_data
            if not game.get("is_completed") and
               game.get("home_team_id") in self.conference_teams and
               game.get("away_team_id") in self.conference_teams
        ]
    
    def _get_team_strength(self, team_id: str) -> Tuple[float, float]:
        """Gets a team's offensive and defensive strength, falling back to league average."""
        stats = self.team_performance.get(team_id)
        if stats and stats.get('games_played', 0) > 0:
            # Use xG if available, otherwise fall back to goals
            attack_metric = stats.get('x_goals_for', stats.get('goals_for', 0))
            defend_metric = stats.get('x_goals_against', stats.get('goals_against', 0))
            games_played = stats['games_played']
            
            # SAFETY GUARDS: Ensure we never divide by zero
            safe_league_avg_xgf = max(self.league_avg_xgf, 0.1)  # Minimum 0.1
            safe_league_avg_xga = max(self.league_avg_xga, 0.1)  # Minimum 0.1
            
            # Calculate per-game metrics
            attack_per_game = attack_metric / games_played if games_played > 0 else safe_league_avg_xgf
            defend_per_game = defend_metric / games_played if games_played > 0 else safe_league_avg_xga
            
            # Calculate strength ratios
            attack_strength = attack_per_game / safe_league_avg_xgf
            defend_strength = defend_per_game / safe_league_avg_xga
            
            # Ensure reasonable bounds (between 0.1 and 5.0)
            attack_strength = max(min(attack_strength, 5.0), 0.1)
            defend_strength = max(min(defend_strength, 5.0), 0.1)
            
            logger.debug(f"Team {team_id} strength for {self.conference} conference: Attack={attack_strength:.2f}, Defense={defend_strength:.2f}")
            return attack_strength, defend_strength
        
        logger.debug(f"Team {team_id} not found or no games played in team_performance for {self.conference} conference, using default strength (1.0, 1.0).")
        return 1.0, 1.0 # Fallback to league average strength

    def _simulate_game(self, game: Dict) -> Tuple[int, int, bool, bool]:
        """
        Simulates a single game.
        Returns:
            Tuple[int, int, bool, bool]:
                - home_goals_reg (int): Goals scored by home team in regulation.
                - away_goals_reg (int): Goals scored by away team in regulation.
                - went_to_shootout (bool): True if the game went to a shootout.
                - home_wins_shootout (bool): True if home team won shootout, False if away team won. Only valid if went_to_shootout is True.
        """
        home_id, away_id = game["home_team_id"], game["away_team_id"]
        
        # Calculate relative team strengths
        home_attack, home_defense = self._get_team_strength(home_id)
        away_attack, away_defense = self._get_team_strength(away_id)
        
        # Calculate expected goals for this matchup
        home_exp_goals = home_attack * away_defense * self.league_avg_xgf
        away_exp_goals = away_attack * home_defense * self.league_avg_xga
        
        # Get result from Poisson distribution
        home_goals_reg = np.random.poisson(home_exp_goals)
        away_goals_reg = np.random.poisson(away_exp_goals)
        
        went_to_shootout = False
        home_wins_shootout = False # Default value

        if home_goals_reg == away_goals_reg:
            went_to_shootout = True
            if np.random.rand() < self.HOME_SHOOTOUT_WIN_PROB:
                home_wins_shootout = True
            else:
                home_wins_shootout = False # Explicitly set, though already default

        logger.debug(f"Simulated game in {self.conference} conference: ID {game.get('id', 'N/A')} ({home_id} vs {away_id}) -> Reg goals: {home_goals_reg}-{away_goals_reg}. Went to SO: {went_to_shootout}. Home won SO: {home_wins_shootout if went_to_shootout else 'N/A'}")
        return home_goals_reg, away_goals_reg, went_to_shootout, home_wins_shootout

    def run_simulations(self, n_simulations: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        Runs the Monte Carlo simulation for n_simulations.
        """

        # Check data quality before running simulations
        warnings = self._check_data_quality()
        if warnings:
            logger.warning("Data quality warnings detected:")
            for warning in warnings:
                logger.warning(f"  {warning}")

        problematic_teams = [w for w in warnings if "Win/loss/draw counts don't add up" in w]
        if problematic_teams:
            # Extract team name from first warning and debug it
            first_warning = problematic_teams[0]
            team_name = first_warning.split("⚠️  ")[1].split(":")[0]
            team_id = next((tid for tid, name in self.team_names.items() if name == team_name), None)
            if team_id:
                self._debug_team_standings(team_id)

        final_ranks = defaultdict(list)
        final_points = defaultdict(list)

        logger.info(f"Starting {n_simulations} Monte Carlo simulations for {self.conference} conference.")
        for sim_idx in range(n_simulations):
            # Log progress e.g. every 10% or every 1000 simulations, whichever is more frequent for large N
            log_progress_interval = 0
            if n_simulations >= 10000:
                log_progress_interval = n_simulations // 100  # Log 100 times
            elif n_simulations >= 100:
                log_progress_interval = n_simulations // 10   # Log 10 times
            elif n_simulations >= 10:
                 log_progress_interval = n_simulations // 5 # Log 5 times
            else:
                log_progress_interval = 1 # Log every sim for small N

            if (sim_idx + 1) % log_progress_interval == 0:
                 logger.debug(f"Running simulation {sim_idx + 1}/{n_simulations} for {self.conference} conference...")

            sim_standings = {team_id: stats.copy() for team_id, stats in self.current_standings.items()}

            for game in self.remaining_games:
                home_id, away_id = game["home_team_id"], game["away_team_id"]
                h_goals_reg, a_goals_reg, went_to_shootout, home_wins_shootout = self._simulate_game(game)

                if went_to_shootout:
                    # Both teams get regulation draw stats (goals, games played, draws count)
                    self._update_regulation_draw(sim_standings[home_id], h_goals_reg, a_goals_reg)
                    self._update_regulation_draw(sim_standings[away_id], a_goals_reg, h_goals_reg)
                    
                    if home_wins_shootout:
                        sim_standings[home_id]["shootout_wins"] += 1
                        sim_standings[home_id]["points"] += 2  # 2 total points for SO win
                        sim_standings[away_id]["points"] += 1   # 1 point for SO loss
                    else: # Away wins shootout
                        sim_standings[away_id]["shootout_wins"] += 1
                        sim_standings[away_id]["points"] += 2  # 2 total points for SO win
                        sim_standings[home_id]["points"] += 1   # 1 point for SO loss
                else:
                    # Regular time result (no shootout)
                    if h_goals_reg > a_goals_reg: # Home win
                        self._update_team_standings(sim_standings[home_id], h_goals_reg, a_goals_reg, "win")
                        self._update_team_standings(sim_standings[away_id], a_goals_reg, h_goals_reg, "loss")
                    elif a_goals_reg > h_goals_reg: # Away win
                        self._update_team_standings(sim_standings[away_id], a_goals_reg, h_goals_reg, "win")
                        self._update_team_standings(sim_standings[home_id], h_goals_reg, a_goals_reg, "loss")
                    # If h_goals_reg == a_goals_reg and not went_to_shootout, it implies an issue or rule change.
                    # This case should not be reached if all draws go to shootout,
                    # but current handling of _update_team_standings for "draw" (which isn't a result type here)
                    # would only update stats, not points. This section assumes decisive results or shootouts.

            # Sort standings and record ranks
            sorted_teams = sorted(sim_standings.values(), key=lambda x: (-x['points'], -x['wins'], -x['goal_difference'], -x['goals_for'], -x['shootout_wins']))
            
            for rank, stats in enumerate(sorted_teams, 1):
                team_id = stats['team_id']
                final_ranks[team_id].append(rank)
                final_points[team_id].append(stats['points'])
        
        logger.info(f"Completed {n_simulations} Monte Carlo simulations for {self.conference} conference.")
        summary_df, qualification_data = self._create_summary_df(final_ranks, final_points)
        
        # This function should return the same signature as the original
        # The third element (rank_dist_df) can be an empty DataFrame if not used.
        return summary_df, final_ranks, pd.DataFrame(), qualification_data

    def _create_summary_df(self, final_ranks: Dict, final_points: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Creates the final summary DataFrame and qualification data dictionary."""
        summary_data = []
        qualification_data = {}

        # Calculate current ranks based on current standings
        current_teams_sorted = sorted(
            self.current_standings.items(), 
            key=lambda x: (
                -x[1]['points'],  # Points (descending)
                -x[1]['wins'],    # Wins (descending)
                -x[1]['goal_difference'],  # Goal difference (descending)
                -x[1]['goals_for'],  # Goals for (descending)
                -x[1].get('shootout_wins', 0)  # Shootout wins (descending)
            )
        )
    
        # Create current rank mapping
        current_rank_map = {
            team_id: rank 
            for rank, (team_id, _) in enumerate(current_teams_sorted, 1)
    }
        
        for team_id, ranks in final_ranks.items():
            current_stats = self.current_standings.get(team_id, {})
            points_list = final_points.get(team_id, [])
            playoff_prob = (np.array(ranks) <= 8).mean() * 100
            
            if ranks:
                avg_rank = np.mean(ranks)
                median_rank = np.median(ranks)
                best_rank = min(ranks)
                worst_rank = max(ranks)
                rank_25 = np.percentile(ranks, 25)
                rank_75 = np.percentile(ranks, 75)
            else:
                avg_rank = median_rank = best_rank = worst_rank = rank_25 = rank_75 = 999

            if points_list:
                avg_points = np.mean(points_list)
                best_points = max(points_list)
                worst_points = min(points_list)
            else:
                avg_points = current_stats.get('points', 0)
                best_points = worst_points = avg_points

            status = ""
            if worst_rank <= 8:
                status = "x-"  # Clinched playoffs
            elif best_rank > 8:
                status = "e-"  # Eliminated from playoffs

            team_name = self.team_names.get(team_id, team_id)
            display_name = f"{status}{team_name}" if status else team_name

            summary_data.append({
                'Team': display_name,
                '_team_id': team_id,
                'Current Points': current_stats.get('points', 0),
                'Current Rank': current_rank_map.get(team_id, 999),
                'Games Played': current_stats.get('games_played', 0),
                'Playoff Qualification %': playoff_prob,
                'Average Final Rank': avg_rank,
                'Average Points': avg_points,
                'Median Final Rank': median_rank,
                'Best Rank': best_rank,
                'Worst Rank': worst_rank,
                'Best Points': best_points,
                'Worst Points': worst_points,
                '_rank_25': rank_25,
                '_rank_75': rank_75,
            })

            qualification_data[team_id] = {
                'games_remaining': len(self.remaining_games),
                'status': status,
                'playoff_probability': playoff_prob,
                'shootout_win_impact': {} # Placeholder for compatibility
            }

        summary_df = pd.DataFrame(summary_data).sort_values(by='Playoff Qualification %', ascending=False).reset_index(drop=True)
        return summary_df, qualification_data

    # Helper methods to update standings state
    def _update_team_standings(self, team_stats: Dict, goals_for: int, goals_against: int, result: str):
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] = team_stats["goals_for"] - team_stats["goals_against"] # Recalculate GD

        if result == "win":
            team_stats["wins"] += 1
            team_stats["points"] += 3
        elif result == "loss":
            team_stats["losses"] += 1
        # No "draw" result type for this method, as draws either go to shootout or are handled by _update_regulation_draw

    def _update_regulation_draw(self, team_stats: Dict, goals_for_in_reg: int, goals_against_in_reg: int):
        """
        Handle regulation draw (used for shootout games).
        Updates games_played, goals, draws count. Points are handled separately.
        """
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for_in_reg
        team_stats["goals_against"] += goals_against_in_reg
        team_stats["goal_difference"] = team_stats["goals_for"] - team_stats["goals_against"] # Recalculate GD
        team_stats["draws"] += 1  # This was a regulation draw
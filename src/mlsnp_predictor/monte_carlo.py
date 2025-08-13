from typing import Dict, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from src.common.utils import time_it
from src.common.BasePredictor import BasePredictor

logger = logging.getLogger(__name__)

class MonteCarloPredictor(BasePredictor):
    """
    Monte Carlo simulation predictor using Poisson distributions.
    Based on expected goals (xG) data for each team.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Monte Carlo predictor."""
        super().__init__(*args, **kwargs)
        
        # Monte Carlo specific constants
        self.HOME_SHOOTOUT_WIN_PROB = 0.55  # Home team has 55% chance in shootout
        
        logger.info(f"Monte Carlo predictor initialized with {len(self.remaining_games)} games to simulate")
    
    def get_method_name(self) -> str:
        """Return the name of this prediction method."""
        return "monte_carlo"
    
    @time_it
    def run_simulations(self, n_simulations: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        Run Monte Carlo simulations for the season.
        
        Args:
            n_simulations: Number of simulations to run
            
        Returns:
            Tuple of summary DataFrame, rank distributions, empty DataFrame, and qualification data
        """
        logger.info(f"Starting {n_simulations:,} Monte Carlo simulations for {self.conference} conference")
        
        # Initialize tracking
        final_ranks = defaultdict(list)
        final_points = defaultdict(list)
        
        # Progress logging
        log_interval = max(1, n_simulations // 10)
        
        # Run simulations
        for sim_idx in range(n_simulations):
            if (sim_idx + 1) % log_interval == 0:
                logger.info(f"Running simulation {sim_idx + 1:,}/{n_simulations:,} ({(sim_idx + 1) / n_simulations * 100:.0f}%)")
            
            # Copy current standings for this simulation
            sim_standings = {
                team_id: stats.copy() 
                for team_id, stats in self.current_standings.items()
            }
            
            # Simulate remaining games
            for game in self.remaining_games:
                home_id = game["home_team_id"]
                away_id = game["away_team_id"]
                
                # Simulate game outcome
                home_goals, away_goals, went_to_shootout, home_wins_shootout = self._simulate_game(game)
                
                # Update standings
                if went_to_shootout:
                    # Both teams get regulation draw stats
                    self._update_regulation_draw(sim_standings[home_id], home_goals, away_goals)
                    self._update_regulation_draw(sim_standings[away_id], away_goals, home_goals)
                    
                    # Award shootout points
                    if home_wins_shootout:
                        sim_standings[home_id]["shootout_wins"] += 1
                        sim_standings[home_id]["points"] += 2
                        sim_standings[away_id]["points"] += 1
                    else: # Away wins shootout
                        sim_standings[away_id]["shootout_wins"] += 1
                        sim_standings[away_id]["points"] += 2
                        sim_standings[home_id]["points"] += 1
                else:
                    # Regular time result
                    if home_goals > away_goals: # Home win
                        self._update_team_standings(sim_standings[home_id], home_goals, away_goals, "win")
                        self._update_team_standings(sim_standings[away_id], away_goals, home_goals, "loss")
                    else:  # Away win
                        self._update_team_standings(sim_standings[away_id], away_goals, home_goals, "win")
                        self._update_team_standings(sim_standings[home_id], home_goals, away_goals, "loss")
            
            # Sort final standings and record ranks
            sorted_teams = sorted(
                sim_standings.values(), 
                key=lambda x: (
                    -x['points'],
                    -x['wins'],
                    -x['goal_difference'],
                    -x['goals_for'],
                    -x['shootout_wins']
                )
            )
            
            for rank, stats in enumerate(sorted_teams, 1):
                team_id = stats['team_id']
                final_ranks[team_id].append(rank)
                final_points[team_id].append(stats['points'])
        
        logger.info(f"Completed {n_simulations} Monte Carlo simulations for {self.conference} conference.")
        
        # Create summary
        summary_df, qualification_data = self._create_summary(final_ranks, final_points)
        
        return summary_df, final_ranks, pd.DataFrame(), qualification_data
    
    def _simulate_game(self, game: Dict) -> Tuple[int, int, bool, bool]:
        """
        Simulate a single game using Poisson distributions.
        
        Args:
            game: Game dictionary with team IDs
            
        Returns:
            Tuple of (home_goals, away_goals, went_to_shootout, home_wins_shootout)
        """
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        
        # Get team strengths
        home_attack, home_defense = self._get_team_strength(home_id)
        away_attack, away_defense = self._get_team_strength(away_id)
        
        # Calculate expected goals
        home_exp_goals = home_attack * away_defense * self.league_avg_xgf
        away_exp_goals = away_attack * home_defense * self.league_avg_xga
        
        # Apply home advantage (10% boost)
        home_exp_goals *= 1.1
        
        # Ensure reasonable bounds
        home_exp_goals = max(0.1, min(5.0, home_exp_goals))
        away_exp_goals = max(0.1, min(5.0, away_exp_goals))
        
        # Sample from Poisson distribution
        home_goals = np.random.poisson(home_exp_goals)
        away_goals = np.random.poisson(away_exp_goals)
        
        # Handle draws with shootout
        went_to_shootout = False
        home_wins_shootout = False
        
        if home_goals == away_goals:
            went_to_shootout = True
            home_wins_shootout = np.random.rand() < self.HOME_SHOOTOUT_WIN_PROB
        
        return home_goals, away_goals, went_to_shootout, home_wins_shootout
    
    def _get_team_strength(self, team_id: str) -> Tuple[float, float]:
        """
        Get team's offensive and defensive strength relative to league average.
        
        Args:
            team_id: Team identifier
            
        Returns:
            Tuple of (attack_strength, defense_strength)
        """
        stats = self.team_performance.get(team_id)
        
        if stats and stats.get('games_played', 0) > 0:
            games_played = stats['games_played']
            
            # Calculate per-game metrics
            attack_per_game = stats.get('x_goals_for', 0) / games_played
            defend_per_game = stats.get('x_goals_against', 0) / games_played
            
            # Calculate relative strengths
            attack_strength = attack_per_game / max(self.league_avg_xgf, 0.1)
            defend_strength = defend_per_game / max(self.league_avg_xga, 0.1)
            
            # Ensure reasonable bounds
            attack_strength = max(0.1, min(5.0, attack_strength))
            defend_strength = max(0.1, min(5.0, defend_strength))
            
            return attack_strength, defend_strength
        
        logger.debug(f"Team {team_id} not found or no games played in team_performance for {self.conference} conference, using default strength (1.0, 1.0).")
        # Default to league average
        return 1.0, 1.0
    
    def _create_summary(self, final_ranks: Dict, final_points: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Create summary DataFrame and qualification data from simulation results.
        
        Args:
            final_ranks: Dict mapping team_id to list of final ranks
            final_points: Dict mapping team_id to list of final points
            
        Returns:
            Tuple of summary DataFrame and qualification data dict
        """
        summary_data = []
        qualification_data = {}
        
        # Get current rankings
        current_rank_map = self.get_current_rank_map()
        
        for team_id in self.conference_teams:
            ranks = final_ranks.get(team_id, [])
            points = final_points.get(team_id, [])
            
            if not ranks:
                logger.warning(f"No simulation data for team {team_id}")
                continue
            
            # Calculate statistics
            ranks_array = np.array(ranks)
            points_array = np.array(points)
            
            playoff_prob = (ranks_array <= 8).mean() * 100
            avg_rank = ranks_array.mean()
            median_rank = np.median(ranks_array)
            best_rank = ranks_array.min()
            worst_rank = ranks_array.max()
            rank_25 = np.percentile(ranks_array, 25)
            rank_75 = np.percentile(ranks_array, 75)
            
            avg_points = points_array.mean()
            best_points = points_array.max()
            worst_points = points_array.min()
            
            # Determine status
            status = ""
            if worst_rank <= 8:
                status = "x-"  # Clinched playoffs
            elif best_rank > 8:
                status = "e-"  # Eliminated from playoffs
            
            current_stats = self.current_standings.get(team_id, {})
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
                'Best Rank': int(best_rank),
                'Worst Rank': int(worst_rank),
                'Best Points': int(best_points),
                'Worst Points': int(worst_points),
                '_rank_25': rank_25,
                '_rank_75': rank_75,
            })
            
            # Qualification data
            games_remaining = sum(1 for game in self.remaining_games 
                                if team_id in [game["home_team_id"], game["away_team_id"]])
            
            qualification_data[team_id] = {
                'games_remaining': games_remaining,
                'status': status,
                'playoff_probability': playoff_prob,
                'shootout_win_impact': {}  # Could calculate impact of additional SO wins
            }
        
        # Create DataFrame and sort by playoff probability
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Playoff Qualification %', ascending=False).reset_index(drop=True)
        
        return summary_df, qualification_data
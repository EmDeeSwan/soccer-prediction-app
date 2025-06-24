from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class BasePredictor(ABC):
    """
    Abstract base class for all season predictors.
    Defines the common interface and shared functionality.
    """
    
    def __init__(self, conference: str, conference_teams: Dict[str, str], 
                 games_data: List[Dict], team_performance: Dict[str, Dict], 
                 league_averages: Dict[str, float]):
        """
        Initialize base predictor with common data.
        
        Args:
            conference: Conference name
            conference_teams: Dict mapping team_id to team_name
            games_data: List of all games (completed and upcoming)
            team_performance: Dict of team performance metrics (xG data)
            league_averages: League-wide averages for xGF and xGA
        """
        self.conference = conference
        self.conference_teams = set(conference_teams.keys())
        self.team_names = conference_teams
        self.games_data = games_data
        self.team_performance = team_performance
        self.league_avg_xgf = league_averages.get('league_avg_xgf', 1.2)
        self.league_avg_xga = league_averages.get('league_avg_xga', 1.2)
        
        # Common data structures
        self.current_standings = self._calculate_current_standings()
        self.remaining_games = self._filter_remaining_games()
        
        logger.info(f"Initialized {self.__class__.__name__} for {conference} conference")
        logger.info(f"Conference teams: {len(self.conference_teams)}")
        logger.info(f"Remaining games: {len(self.remaining_games)}")
    
    @abstractmethod
    def run_simulations(self, n_simulations: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        Run predictions/simulations and return results.
        
        Args:
            n_simulations: Number of simulations to run (or ignored for ML)
            
        Returns:
            Tuple of:
            - summary_df: DataFrame with team predictions
            - simulation_results: Dict of detailed results (optional)
            - rank_dist_df: DataFrame with rank distributions (optional)
            - qualification_data: Dict with qualification info
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of this prediction method."""
        pass
    
    def _calculate_current_standings(self) -> Dict[str, Dict]:
        """Calculate current standings from completed games."""
        standings = defaultdict(lambda: {
            "team_id": None, "name": "", "points": 0, "goal_difference": 0,
            "games_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "shootout_wins": 0
        })
        
        # Initialize all conference teams
        for team_id in self.conference_teams:
            standings[team_id]["team_id"] = team_id
            standings[team_id]["name"] = self.team_names.get(team_id, f"Team {team_id}")
        
        # Process completed games
        for game in self.games_data:
            if not game.get("is_completed"):
                continue
                
            home_id, away_id = game["home_team_id"], game["away_team_id"]
            
            # Skip if not both conference teams
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue
            
            try:
                home_score = int(game.get("home_score", 0))
                away_score = int(game.get("away_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid scores for game {game.get('game_id', 'unknown')}")
                continue
            
            # Process game result
            if game.get("went_to_shootout"):
                # Both teams get a draw
                self._update_regulation_draw(standings[home_id], home_score, away_score)
                self._update_regulation_draw(standings[away_id], away_score, home_score)
                
                # Determine shootout winner
                home_pens = game.get("home_penalties", 0) or 0
                away_pens = game.get("away_penalties", 0) or 0
                
                if home_pens > away_pens:
                    standings[home_id]["shootout_wins"] += 1
                    standings[home_id]["points"] += 2  # Total 2 points for SO win
                    standings[away_id]["points"] += 1   # 1 point for SO loss
                else:
                    standings[away_id]["shootout_wins"] += 1
                    standings[away_id]["points"] += 2
                    standings[home_id]["points"] += 1
            else:
                # Regular time result
                if home_score > away_score:
                    self._update_team_standings(standings[home_id], home_score, away_score, "win")
                    self._update_team_standings(standings[away_id], away_score, home_score, "loss")
                elif away_score > home_score:
                    self._update_team_standings(standings[away_id], away_score, home_score, "win")
                    self._update_team_standings(standings[home_id], home_score, away_score, "loss")
                else:
                    # Regular draw (shouldn't happen in MLS Next Pro)
                    logger.warning(f"Regular draw in game {game.get('game_id', 'unknown')}")
                    self._update_regulation_draw(standings[home_id], home_score, away_score)
                    self._update_regulation_draw(standings[away_id], away_score, home_score)
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1
        
        # Calculate goal differences
        for team_id, stats in standings.items():
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
        
        return {team_id: dict(stats) for team_id, stats in standings.items()}
    
    def _filter_remaining_games(self) -> List[Dict]:
        """Filter for future games to be simulated."""
        return [
            game for game in self.games_data
            if not game.get("is_completed") and
               game.get("home_team_id") in self.conference_teams and
               game.get("away_team_id") in self.conference_teams
        ]
    
    def _update_team_standings(self, team_stats: Dict, goals_for: int, goals_against: int, result: str):
        """Update team standings for a regular time result."""
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] = team_stats["goals_for"] - team_stats["goals_against"]
        
        if result == "win":
            team_stats["wins"] += 1
            team_stats["points"] += 3
        elif result == "loss":
            team_stats["losses"] += 1
    
    def _update_regulation_draw(self, team_stats: Dict, goals_for: int, goals_against: int):
        """Update team standings for a regulation draw (goes to shootout)."""
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] = team_stats["goals_for"] - team_stats["goals_against"]
        team_stats["draws"] += 1
    
    def get_current_rank_map(self) -> Dict[str, int]:
        """Get current rankings based on standings."""
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
        
        return {
            team_id: rank 
            for rank, (team_id, _) in enumerate(current_teams_sorted, 1)
        }
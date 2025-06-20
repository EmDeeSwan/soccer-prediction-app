from pydantic import BaseModel
from typing import Optional, List, Dict, Any, ClassVar, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from scipy import stats
import json
import shutil
from src.common.utils import time_it
from src.common.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class SimulationRequest(BaseModel):
    conference: str  # "eastern" or "western"
    n_simulations: int = 25000
    include_playoffs: bool = False
    simulation_preset: Optional[str] = "standard"
    SIMULATION_PRESETS: ClassVar[Dict[str, Dict[str, any]]] = {
        "quick": {
            "count": 1000,
            "description": "Quick estimate (~3 seconds)",
            "accuracy": "±3% margin of error"
        },
        "standard": {
            "count": 25000,
            "description": "Professional accuracy (~1 minute)",
            "accuracy": "±0.6% margin of error"
        },
        "detailed": {
            "count": 50000,
            "description": "High precision (~2 minutes)",
            "accuracy": "±0.4% margin of error"
        },
        "research": {
            "count": 100000,
            "description": "Maximum precision (~4 minutes)",
            "accuracy": "±0.3% margin of error"
        }
    }
    
class PlayoffSeedingRequest(BaseModel):
    """Request model for custom playoff seeding"""
    eastern_seeds: Dict[int, str]  # {1: "team_id", 2: "team_id", ...}
    western_seeds: Dict[int, str]
    n_simulations: int = 10000

class TeamPerformance(BaseModel):
    """Model for team performance data"""
    team_id: str
    team_name: str
    current_points: int
    games_played: int
    playoff_probability: float
    average_final_rank: float
    
class SimulationResponse(BaseModel):
    """Response model for simulation results"""
    simulation_id: str
    conference: str
    status: str  # "running", "completed", "failed"
    regular_season_complete: bool
    playoff_simulation_available: bool
    results: Optional[List[TeamPerformance]]
    
class PlayoffBracket(BaseModel):
    """Model for playoff bracket structure"""
    round_1: List[Dict[str, Any]]
    round_2: List[Dict[str, Any]]
    conference_final: Dict[str, Any]
    championship: Optional[Dict[str, Any]]

class LoginCredentials(BaseModel):
    username_or_email: str
    password: str


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
    


"""
Machine Learning predictor for MLS Next Pro regular season.
Uses AutoML (AutoGluon/sklearn) to predict game outcomes directly.
"""
# AutoML imports with fallback
try:
    from autogluon.tabular import TabularPredictor
    AUTOML_AVAILABLE = True
    AUTOML_LIBRARY = "autogluon"
except ImportError:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        AUTOML_AVAILABLE = True
        AUTOML_LIBRARY = "sklearn"
    except ImportError:
        AUTOML_AVAILABLE = False
        AUTOML_LIBRARY = None

class MLPredictor(BasePredictor):    
    def __init__(self, *args, model_path: str = None, **kwargs):
        """
        Initialize ML predictor.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        super().__init__(*args, **kwargs)
        
        self.use_automl = AUTOML_AVAILABLE
        self.ml_model = None
        self.model_version = None
        self._feature_names = None
        
        # Set model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            # Use Railway volume path if available
            railway_volume = Path("/app/models")
            if railway_volume.exists():
                self.model_path = railway_volume / f"mlsnp_{self.conference}_{datetime.now().strftime('%Y%m')}.pkl"
            else:
                # Local fallback
                self.model_path = Path(f"models/mlsnp_{self.conference}_{datetime.now().strftime('%Y%m')}.pkl")
        
        # Ensure model directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pre-compute features for efficiency
        self._precompute_features()
        
        # Initialize or load model
        if self.use_automl:
            self._initialize_ml_model()
        else:
            logger.warning("AutoML not available, predictions will fall back to traditional method")
    
    def get_method_name(self) -> str:
        """Return the name of this prediction method."""
        return "machine_learning"
    
    def _precompute_features(self):
        """Pre-compute expensive features to avoid recalculating."""
        logger.info("Pre-computing features for ML optimization...")
        
        self.completed_games = [g for g in self.games_data if g.get('is_completed')]
        self.team_form_cache = {}
        self.team_h2h_cache = {}
        self.team_rest_cache = {}
        
        current_date = datetime.now()
        
        for team_id in self.conference_teams:
            # Calculate form for each team
            self.team_form_cache[team_id] = self._calculate_form(team_id, current_date)
            
            # Calculate days since last game
            self.team_rest_cache[team_id] = self._calculate_rest_days(team_id, current_date)
            
            # Pre-calculate H2H records
            self.team_h2h_cache[team_id] = {}
            for opponent_id in self.conference_teams:
                if opponent_id != team_id:
                    self.team_h2h_cache[team_id][opponent_id] = self._calculate_h2h(
                        team_id, opponent_id
                    )
        
        logger.info(f"Pre-computed features for {len(self.conference_teams)} teams")
    
    def _calculate_form(self, team_id: str, before_date: datetime, n_games: int = 5) -> Dict[str, float]:
        """Calculate team form from recent games."""
        recent_games = []
        
        # Get recent games before the specified date
        for game in reversed(self.completed_games):
            # Skip future games relative to before_date
            game_date = game.get('date')
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            if game_date and game_date >= before_date:
                continue
                
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                recent_games.append(game)
            if len(recent_games) >= n_games:
                break
        
        if not recent_games:
            return {
                'points_per_game': 1.0,
                'goals_for_per_game': self.league_avg_xgf,
                'goals_against_per_game': self.league_avg_xga
            }
        
        total_points = total_gf = total_ga = 0
        
        for game in recent_games:
            if game['home_team_id'] == team_id:
                gf, ga = game.get('home_score', 0), game.get('away_score', 0)
            else:
                gf, ga = game.get('away_score', 0), game.get('home_score', 0)
            
            total_gf += gf
            total_ga += ga
            
            if game.get('went_to_shootout'):
                # Determine if team won shootout
                if game['home_team_id'] == team_id:
                    won_shootout = (game.get('home_penalties', 0) > game.get('away_penalties', 0))
                else:
                    won_shootout = (game.get('away_penalties', 0) > game.get('home_penalties', 0))
                total_points += 2 if won_shootout else 1
            else:
                if gf > ga:
                    total_points += 3
                elif gf == ga:
                    total_points += 1
        
        n = len(recent_games)
        return {
            'points_per_game': total_points / n,
            'goals_for_per_game': total_gf / n,
            'goals_against_per_game': total_ga / n
        }
    
    def _calculate_h2h(self, team_id: str, opponent_id: str) -> Dict[str, float]:
        """Calculate head-to-head record between two teams."""
        h2h_games = []
        
        for game in self.completed_games:
            if ((game['home_team_id'] == team_id and game['away_team_id'] == opponent_id) or
                (game['home_team_id'] == opponent_id and game['away_team_id'] == team_id)):
                h2h_games.append(game)
        
        if not h2h_games:
            return {
                'h2h_games_played': 0.0,
                'h2h_win_rate': 0.5,
                'h2h_goals_for_avg': self.league_avg_xgf,
                'h2h_goals_against_avg': self.league_avg_xga
            }
        
        wins = total_gf = total_ga = 0
        
        for game in h2h_games:
            if game['home_team_id'] == team_id:
                gf, ga = game.get('home_score', 0), game.get('away_score', 0)
            else:
                gf, ga = game.get('away_score', 0), game.get('home_score', 0)
            
            total_gf += gf
            total_ga += ga
            
            if gf > ga:
                wins += 1
            elif game.get('went_to_shootout'):
                if game['home_team_id'] == team_id:
                    if game.get('home_penalties', 0) > game.get('away_penalties', 0):
                        wins += 0.5  # Count shootout win as half
                else:
                    if game.get('away_penalties', 0) > game.get('home_penalties', 0):
                        wins += 0.5
        
        n = len(h2h_games)
        return {
            'h2h_games_played': float(n),
            'h2h_win_rate': wins / n,
            'h2h_goals_for_avg': total_gf / n,
            'h2h_goals_against_avg': total_ga / n
        }
    
    def _calculate_rest_days(self, team_id: str, before_date: datetime) -> float:
        """Calculate days since team's last game."""
        for game in reversed(self.completed_games):
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                game_date = game.get('date')
                if isinstance(game_date, str):
                    game_date = pd.to_datetime(game_date)
                    
                if game_date and game_date < before_date:
                    days_rest = (before_date - game_date).days
                    return min(days_rest, 14)  # Cap at 14 days
        
        return 7.0  # Default if no previous games
    
    def _extract_features(self, team_id: str, opponent_id: str, is_home: bool, 
                         game_date: Optional[datetime] = None, 
                         games_before: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Extract features for ML model prediction.
        Bug fix: Proper date handling and historical data usage.
        """
        features = {}
        
        # Basic features
        features['is_home'] = 1.0 if is_home else 0.0
        
        # Team strength features from xG data
        team_stats = self.team_performance.get(team_id, {})
        opp_stats = self.team_performance.get(opponent_id, {})
        
        # Offensive and defensive strength
        team_games = max(team_stats.get('games_played', 1), 1)
        opp_games = max(opp_stats.get('games_played', 1), 1)
        
        features['team_xgf_per_game'] = team_stats.get('x_goals_for', 0) / team_games
        features['team_xga_per_game'] = team_stats.get('x_goals_against', 0) / team_games
        features['opp_xgf_per_game'] = opp_stats.get('x_goals_for', 0) / opp_games
        features['opp_xga_per_game'] = opp_stats.get('x_goals_against', 0) / opp_games
        
        # Relative strength features
        features['xg_diff'] = features['team_xgf_per_game'] - features['team_xga_per_game']
        features['opp_xg_diff'] = features['opp_xgf_per_game'] - features['opp_xga_per_game']
        
        # Use pre-computed or calculate form/H2H
        if hasattr(self, 'team_form_cache') and not game_date:
            # Use pre-computed values for predictions
            team_form = self.team_form_cache.get(team_id, {
                'points_per_game': 1.0,
                'goals_for_per_game': self.league_avg_xgf,
                'goals_against_per_game': self.league_avg_xga
            })
            opp_form = self.team_form_cache.get(opponent_id, {
                'points_per_game': 1.0,
                'goals_for_per_game': self.league_avg_xgf,
                'goals_against_per_game': self.league_avg_xga
            })
            
            h2h = self.team_h2h_cache.get(team_id, {}).get(opponent_id, {
                'h2h_games_played': 0.0,
                'h2h_win_rate': 0.5,
                'h2h_goals_for_avg': self.league_avg_xgf,
                'h2h_goals_against_avg': self.league_avg_xga
            })
            
            features['team_rest_days'] = self.team_rest_cache.get(team_id, 7.0)
            features['opp_rest_days'] = self.team_rest_cache.get(opponent_id, 7.0)
        else:
            # Calculate for historical training data
            if not game_date:
                game_date = datetime.now()
            
            # Use only games before this date
            if games_before is not None:
                team_form = self._calculate_form_from_games(team_id, game_date, games_before)
                opp_form = self._calculate_form_from_games(opponent_id, game_date, games_before)
                h2h = self._calculate_h2h_from_games(team_id, opponent_id, games_before)
                features['team_rest_days'] = self._calculate_rest_days_from_games(team_id, game_date, games_before)
                features['opp_rest_days'] = self._calculate_rest_days_from_games(opponent_id, game_date, games_before)
            else:
                # Fallback to cached values
                team_form = {'points_per_game': 1.0, 'goals_for_per_game': self.league_avg_xgf, 'goals_against_per_game': self.league_avg_xga}
                opp_form = {'points_per_game': 1.0, 'goals_for_per_game': self.league_avg_xgf, 'goals_against_per_game': self.league_avg_xga}
                h2h = {'h2h_games_played': 0.0, 'h2h_win_rate': 0.5, 'h2h_goals_for_avg': self.league_avg_xgf, 'h2h_goals_against_avg': self.league_avg_xga}
                features['team_rest_days'] = 7.0
                features['opp_rest_days'] = 7.0
        
        # Add form features
        features.update({
            'team_form_points': team_form['points_per_game'],
            'team_form_gf': team_form['goals_for_per_game'],
            'team_form_ga': team_form['goals_against_per_game'],
            'opp_form_points': opp_form['points_per_game'],
            'opp_form_gf': opp_form['goals_for_per_game'],
            'opp_form_ga': opp_form['goals_against_per_game']
        })
        
        # Add H2H features
        features.update(h2h)
        
        # Temporal features
        if isinstance(game_date, datetime):
            features['month'] = game_date.month
            features['day_of_week'] = game_date.weekday()
            features['is_weekend'] = 1.0 if game_date.weekday() >= 5 else 0.0
        else:
            # Default temporal features
            features['month'] = 6
            features['day_of_week'] = 3
            features['is_weekend'] = 0.0
        
        return features
    
    def _calculate_form_from_games(self, team_id: str, before_date: datetime, games: List[Dict]) -> Dict[str, float]:
        """Calculate form using only specified games."""
        recent_games = []
        
        for game in reversed(games):
            game_date = game.get('date')
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            if game_date >= before_date:
                continue
                
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                recent_games.append(game)
            if len(recent_games) >= 5:
                break
        
        if not recent_games:
            return {
                'points_per_game': 1.0,
                'goals_for_per_game': self.league_avg_xgf,
                'goals_against_per_game': self.league_avg_xga
            }
        
        # Calculate stats from recent games
        total_points = total_gf = total_ga = 0
        
        for game in recent_games:
            if game['home_team_id'] == team_id:
                gf, ga = game.get('home_score', 0), game.get('away_score', 0)
            else:
                gf, ga = game.get('away_score', 0), game.get('home_score', 0)
            
            total_gf += gf
            total_ga += ga
            
            if game.get('went_to_shootout'):
                if game['home_team_id'] == team_id:
                    won_shootout = (game.get('home_penalties', 0) > game.get('away_penalties', 0))
                else:
                    won_shootout = (game.get('away_penalties', 0) > game.get('home_penalties', 0))
                total_points += 2 if won_shootout else 1
            else:
                if gf > ga:
                    total_points += 3
                elif gf == ga:
                    total_points += 1
        
        n = len(recent_games)
        return {
            'points_per_game': total_points / n,
            'goals_for_per_game': total_gf / n,
            'goals_against_per_game': total_ga / n
        }
    
    def _calculate_h2h_from_games(self, team_id: str, opponent_id: str, games: List[Dict]) -> Dict[str, float]:
        """Calculate H2H using only specified games."""
        h2h_games = []
        
        for game in games:
            if ((game['home_team_id'] == team_id and game['away_team_id'] == opponent_id) or
                (game['home_team_id'] == opponent_id and game['away_team_id'] == team_id)):
                h2h_games.append(game)
        
        if not h2h_games:
            return {
                'h2h_games_played': 0.0,
                'h2h_win_rate': 0.5,
                'h2h_goals_for_avg': self.league_avg_xgf,
                'h2h_goals_against_avg': self.league_avg_xga
            }
        
        wins = total_gf = total_ga = 0
        
        for game in h2h_games:
            if game['home_team_id'] == team_id:
                gf, ga = game.get('home_score', 0), game.get('away_score', 0)
            else:
                gf, ga = game.get('away_score', 0), game.get('home_score', 0)
            
            total_gf += gf
            total_ga += ga
            
            if gf > ga:
                wins += 1
            elif game.get('went_to_shootout'):
                if game['home_team_id'] == team_id:
                    if game.get('home_penalties', 0) > game.get('away_penalties', 0):
                        wins += 0.5
                else:
                    if game.get('away_penalties', 0) > game.get('home_penalties', 0):
                        wins += 0.5
        
        n = len(h2h_games)
        return {
            'h2h_games_played': float(n),
            'h2h_win_rate': wins / n,
            'h2h_goals_for_avg': total_gf / n,
            'h2h_goals_against_avg': total_ga / n
        }
    
    def _calculate_rest_days_from_games(self, team_id: str, before_date: datetime, games: List[Dict]) -> float:
        """Calculate rest days using only specified games."""
        for game in reversed(games):
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                game_date = game.get('date')
                if isinstance(game_date, str):
                    game_date = pd.to_datetime(game_date)
                    
                if game_date and game_date < before_date:
                    days_rest = (before_date - game_date).days
                    return min(days_rest, 14)
        
        return 7.0
    
    def _initialize_ml_model(self):
        """Initialize or load the AutoML model."""
        # For AutoGluon, check the actual directory path, not the .pkl path
        if AUTOML_LIBRARY == "autogluon":
            autogluon_path = self.model_path.parent / f"ag_{self.conference}"
            if autogluon_path.exists():
                try:
                    self.ml_model = TabularPredictor.load(str(autogluon_path))
                    self.model_version = f"ag_{self.conference}"
                    logger.info(f"Loaded existing AutoGluon model: {self.model_version}")
                    return
                except Exception as e:
                    logger.warning(f"Could not load existing AutoGluon model: {e}")
        elif self.model_path.exists():  # sklearn logic
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_model = model_data['model']
                    self._feature_names = model_data['feature_names']
                    self.model_version = model_data.get('version', self.model_path.stem)
                logger.info(f"Loaded existing sklearn model: {self.model_version}")
                return
            except Exception as e:
                logger.warning(f"Could not load existing sklearn model: {e}")
        
        # No model exists - will need to train
        logger.info("No pre-trained model found. Call train_model() to train a new model.")
    
    @time_it
    def train_model(self, time_limit: int = 300) -> bool:
        """
        Train a new ML model using historical data.
        Bug fix: Prevents data leakage by using only past games for features.
        
        Args:
            time_limit: Time limit in seconds for AutoGluon training
            
        Returns:
            True if training successful, False otherwise
        """
        if not AUTOML_AVAILABLE:  # Check the global flag instead
            logger.error("No ML libraries available. Cannot train model.")
            return False
        
        logger.info(f"Training new {AUTOML_LIBRARY} model...")
        training_data = self._prepare_training_data()
        
        if len(training_data) < 50:
            logger.warning("Insufficient training data for ML model.")
            return False
        
        try:
            if AUTOML_LIBRARY == "autogluon":
                self.ml_model = TabularPredictor(
                    label='goals',
                    path=str(self.model_path.parent / f"ag_{self.conference}"),
                    problem_type='regression',
                    eval_metric='root_mean_squared_error'
                )
                
                self.ml_model.fit(
                    training_data,
                    time_limit=time_limit,
                    presets='best_quality',
                    verbosity=0
                )
                
                # Save model path
                self.model_version = f"ag_{self.conference}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"AutoGluon model trained: {self.model_version}")
                
            elif AUTOML_LIBRARY == "sklearn":
                self._feature_names = [col for col in training_data.columns if col != 'goals']
                X = training_data[self._feature_names]
                y = training_data['goals']
                
                self.ml_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                self.ml_model.fit(X, y)
                
                # Save model with metadata
                self.model_version = f"sklearn_{self.conference}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_data = {
                    'model': self.ml_model,
                    'feature_names': self._feature_names,
                    'version': self.model_version,
                    'training_date': datetime.now().isoformat(),
                    'training_samples': len(X)
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                logger.info(f"Sklearn model trained and saved: {self.model_version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare training data with proper temporal ordering.
        Bug fix: Prevents data leakage by using only historical games.
        """
        training_records = []
        
        # Sort games by date to ensure temporal ordering
        sorted_games = sorted(self.completed_games, key=lambda x: x.get('date', ''))
        
        # Need minimum games for meaningful features
        if len(sorted_games) < 10:
            logger.warning("Not enough completed games for training")
            return pd.DataFrame()
        
        for i, game in enumerate(sorted_games):
            # Skip early games without enough history
            if i < 5:
                continue
            
            # Only use games BEFORE this one for feature calculation
            games_before = sorted_games[:i]
            
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            
            # Skip if not both conference teams
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue
            
            game_date = game.get('date')
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            
            # Extract features using only historical data
            home_features = self._extract_features(
                home_id, away_id, True, game_date, games_before
            )
            home_features['goals'] = game.get('home_score', 0)
            training_records.append(home_features)
            
            away_features = self._extract_features(
                away_id, home_id, False, game_date, games_before
            )
            away_features['goals'] = game.get('away_score', 0)
            training_records.append(away_features)
        
        logger.info(f"Prepared {len(training_records)} training samples")
        return pd.DataFrame(training_records)
    
    @time_it
    def run_simulations(self, n_simulations: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        Run ML predictions (no simulations needed).
        
        Args:
            n_simulations: Ignored for ML predictions
            
        Returns:
            Tuple of summary DataFrame, empty dict, empty DataFrame, and qualification data
        """
        logger.info(f"Running ML predictions for {self.conference} conference")
        
        if not self.ml_model:
            logger.warning("No ML model available. Train a model first or fall back to Monte Carlo.")
            return self._run_fallback_predictions()
        
        # Predict all remaining games
        game_predictions = []
        
        for i, game in enumerate(self.remaining_games):
            if i % 20 == 0 and i > 0:
                logger.info(f"Predicting game {i}/{len(self.remaining_games)}...")
            
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]
            
            # Get features
            home_features = self._extract_features(home_id, away_id, True)
            away_features = self._extract_features(away_id, home_id, False)
            
            # Make predictions
            if AUTOML_LIBRARY == "autogluon":
                home_pred = self.ml_model.predict(pd.DataFrame([home_features]))[0]
                away_pred = self.ml_model.predict(pd.DataFrame([away_features]))[0]
            else:  # sklearn
                home_X = pd.DataFrame([home_features])[self._feature_names]
                away_X = pd.DataFrame([away_features])[self._feature_names]
                
                home_pred = self.ml_model.predict(home_X)[0]
                away_pred = self.ml_model.predict(away_X)[0]
            
            # Ensure positive predictions
            home_pred = max(0.1, home_pred)
            away_pred = max(0.1, away_pred)
            
            # Calculate win probabilities
            home_win_prob = self._calculate_win_probability(home_pred, away_pred)
            away_win_prob = self._calculate_win_probability(away_pred, home_pred)
            draw_prob = 1.0 - home_win_prob - away_win_prob
            
            game_predictions.append({
                'game': game,
                'home_pred': home_pred,
                'away_pred': away_pred,
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob,
                'draw_prob': draw_prob
            })
        
        # Calculate expected points
        expected_points = self._calculate_expected_points(game_predictions)
        
        # Calculate playoff probabilities
        team_projections = self._create_projections(expected_points, game_predictions)
        playoff_probs = self._calculate_playoff_probabilities(team_projections)
        
        # Create summary
        summary_df, qualification_data = self._create_summary(team_projections, playoff_probs)
        
        return summary_df, {}, pd.DataFrame(), qualification_data
    
    def _calculate_win_probability(self, team_goals: float, opp_goals: float) -> float:
        """Calculate win probability using Poisson distributions."""
        max_goals = 8 # Reasonable upper bound for soccer
        win_prob = 0
        
        for home_score in range(max_goals):
            for away_score in range(max_goals):
                home_prob = stats.poisson.pmf(home_score, team_goals)
                away_prob = stats.poisson.pmf(away_score, opp_goals)
                game_prob = home_prob * away_prob
                
                if home_score > away_score:
                    win_prob += game_prob
                elif home_score == away_score:
                    # Shootout probability
                    shootout_advantage = 0.55 if team_goals >= opp_goals else 0.45
                    win_prob += game_prob * shootout_advantage
        
        return min(win_prob, 0.95)
    
    def _calculate_expected_points(self, game_predictions: List[Dict]) -> Dict[str, float]:
        """Calculate expected points for each team."""
        expected_points = {}
        
        for team_id in self.conference_teams:
            current_points = self.current_standings.get(team_id, {}).get('points', 0)
            expected_additional = 0
            
            for pred in game_predictions:
                game = pred['game']
                if game["home_team_id"] == team_id:
                    expected_additional += (
                        pred['home_win_prob'] * 3 +
                        pred['draw_prob'] * 1.5  # Average shootout points
                    )
                elif game["away_team_id"] == team_id:
                    expected_additional += (
                        pred['away_win_prob'] * 3 +
                        pred['draw_prob'] * 1.5
                    )
            
            expected_points[team_id] = current_points + expected_additional
        
        return expected_points
    
    def _create_projections(self, expected_points: Dict[str, float], 
                           game_predictions: List[Dict]) -> List[Dict]:
        """Create team projections with uncertainty estimates."""
        projections = []
        
        for team_id, exp_points in expected_points.items():
            current_stats = self.current_standings.get(team_id, {})
            
            # Count remaining games
            games_remaining = sum(1 for pred in game_predictions 
                                if team_id in [pred['game']["home_team_id"], 
                                            pred['game']["away_team_id"]])
            
            # Estimate uncertainty
            points_std = np.sqrt(games_remaining) * 0.8
            
            projections.append({
                'team_id': team_id,
                'team_name': self.team_names.get(team_id, team_id),
                'current_points': current_stats.get('points', 0),
                'games_played': current_stats.get('games_played', 0),
                'expected_points': exp_points,
                'points_std': points_std,
                'games_remaining': games_remaining
            })
        
        return sorted(projections, key=lambda x: x['expected_points'], reverse=True)
    
    def _calculate_playoff_probabilities(self, projections: List[Dict], 
                                       n_samples: int = 10000) -> Dict[str, float]:
        """Calculate playoff probabilities using point distributions."""
        playoff_counts = defaultdict(int)
        
        for _ in range(n_samples):
            # Sample points from normal distributions
            sample_points = []
            for proj in projections:
                sampled = np.random.normal(proj['expected_points'], proj['points_std'])
                sample_points.append((proj['team_id'], max(0, sampled)))
            
            # Top 8 make playoffs
            sample_points.sort(key=lambda x: x[1], reverse=True)
            playoff_teams = [team_id for team_id, _ in sample_points[:8]]
            
            for team_id in playoff_teams:
                playoff_counts[team_id] += 1
        
        # Convert to probabilities
        return {team_id: count / n_samples for team_id, count in playoff_counts.items()}
    
    def _create_summary(self, projections: List[Dict], 
                       playoff_probs: Dict[str, float]) -> Tuple[pd.DataFrame, Dict]:
        """Create summary DataFrame from projections."""
        summary_data = []
        qualification_data = {}
        
        current_rank_map = self.get_current_rank_map()
        
        for i, proj in enumerate(projections):
            team_id = proj['team_id']
            playoff_prob = playoff_probs.get(team_id, 0) * 100
            
            # For ML predictions, we don't have Monte Carlo rank distributions
            # So we'll use the expected rank based on expected points
            expected_rank = i + 1
            
            summary_data.append({
                'Team': proj['team_name'],
                '_team_id': team_id,
                'Current Points': proj['current_points'],
                'Current Rank': current_rank_map.get(team_id, 999),
                'Games Played': proj['games_played'],
                'Playoff Qualification %': playoff_prob,
                'Average Final Rank': expected_rank,
                'Average Points': proj['expected_points'],
                'Median Final Rank': expected_rank,
                'Best Rank': max(1, expected_rank - 2),  # Rough estimate
                'Worst Rank': min(len(projections), expected_rank + 2),
                'Best Points': int(proj['expected_points'] + 2 * proj['points_std']),
                'Worst Points': int(max(0, proj['expected_points'] - 2 * proj['points_std'])),
                '_rank_25': max(1, expected_rank - 1),
                '_rank_75': min(len(projections), expected_rank + 1),
            })
            
            qualification_data[team_id] = {
                'games_remaining': proj['games_remaining'],
                'status': '',
                'playoff_probability': playoff_prob,
                'shootout_win_impact': {}
            }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Playoff Qualification %', ascending=False).reset_index(drop=True)
        
        return summary_df, qualification_data
    
    def _run_fallback_predictions(self) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """Fallback predictions when no ML model is available."""
        logger.warning("Running fallback predictions based on current form")
        
        # Simple linear projection based on current points per game
        summary_data = []
        qualification_data = {}
        
        for team_id in self.conference_teams:
            current_stats = self.current_standings.get(team_id, {})
            games_played = max(1, current_stats.get('games_played', 1))
            current_points = current_stats.get('points', 0)
            
            # Calculate points per game
            ppg = current_points / games_played
            
            # Count remaining games
            games_remaining = sum(1 for game in self.remaining_games 
                                if team_id in [game["home_team_id"], game["away_team_id"]])
            
            # Project final points
            projected_points = current_points + (ppg * games_remaining)
            
            summary_data.append({
                'Team': self.team_names.get(team_id, team_id),
                '_team_id': team_id,
                'Current Points': current_points,
                'Current Rank': 0,  # Will calculate after
                'Games Played': games_played,
                'Playoff Qualification %': 50.0,  # Placeholder
                'Average Final Rank': 0,  # Will calculate after
                'Average Points': projected_points,
                'Median Final Rank': 0,
                'Best Rank': 1,
                'Worst Rank': len(self.conference_teams),
                'Best Points': int(projected_points * 1.2),
                'Worst Points': int(projected_points * 0.8),
                '_rank_25': 0,
                '_rank_75': 0,
            })
            
            qualification_data[team_id] = {
                'games_remaining': games_remaining,
                'status': '',
                'playoff_probability': 50.0,
                'shootout_win_impact': {}
            }
        
        # Sort and assign ranks
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Average Points', ascending=False).reset_index(drop=True)
        
        # Update ranks and playoff probabilities
        for idx, row in summary_df.iterrows():
            rank = idx + 1
            summary_df.at[idx, 'Average Final Rank'] = rank
            summary_df.at[idx, 'Median Final Rank'] = rank
            summary_df.at[idx, '_rank_25'] = max(1, rank - 1)
            summary_df.at[idx, '_rank_75'] = min(len(summary_df), rank + 1)
            
            # Simple playoff probability based on rank
            if rank <= 6:
                playoff_prob = 90.0
            elif rank <= 8:
                playoff_prob = 70.0
            elif rank <= 10:
                playoff_prob = 30.0
            else:
                playoff_prob = 10.0
            
            summary_df.at[idx, 'Playoff Qualification %'] = playoff_prob
        
        return summary_df, {}, pd.DataFrame(), qualification_data
    
class MLModelManager:
    """
    Manages ML model lifecycle including training, versioning, and storage.
    """
    
    def __init__(self, db_manager: DatabaseManager, model_base_path: str = None):
        """
        Initialize model manager.
        
        Args:
            db_manager: Database manager instance
            model_base_path: Base path for model storage (defaults to Railway volume or local)
        """
        self.db_manager = db_manager
        
        # Set model storage path
        if model_base_path:
            self.model_base_path = Path(model_base_path)
        else:
            # Check for Railway volume
            railway_volume = Path("/app/models")
            if railway_volume.exists():
                self.model_base_path = railway_volume
                logger.info(f"Using Railway volume for models: {railway_volume}")
            else:
                # Local fallback
                self.model_base_path = Path("models")
                logger.info(f"Using local directory for models: {self.model_base_path}")
        
        # Ensure directory exists
        self.model_base_path.mkdir(parents=True, exist_ok=True)
    
    async def register_model(
        self,
        model_name: str,
        conference: str,
        version: str,
        file_path: str,
        training_games_count: int,
        performance_metrics: Dict[str, float]
    ) -> int:
        """
        Register a trained model in the database.
        
        Args:
            model_name: Name of the model
            conference: Conference the model was trained for
            version: Model version string
            file_path: Path where model is stored
            training_games_count: Number of games used for training
            performance_metrics: Dict of performance metrics (RMSE, MAE, etc.)
            
        Returns:
            model_id from database
        """
        query = """
            INSERT INTO ml_models (
                model_name, conference, version, file_path,
                training_games_count, performance_metrics
            ) VALUES (
                :model_name, :conference, :version, :file_path,
                :training_games_count, :performance_metrics
            )
            ON CONFLICT (conference, version) DO UPDATE SET
                performance_metrics = EXCLUDED.performance_metrics,
                training_date = NOW()
            RETURNING model_id
        """
        
        values = {
            "model_name": model_name,
            "conference": conference,
            "version": version,
            "file_path": file_path,
            "training_games_count": training_games_count,
            "performance_metrics": json.dumps(performance_metrics)
        }
        
        result = await self.db_manager.db.fetch_one(query, values=values)
        logger.info(f"Registered model {model_name} v{version} with ID: {result['model_id']}")
        return result['model_id']
    
    async def get_latest_model(self, conference: str) -> Optional[Dict]:
        """
        Get the latest active model for a conference.
        
        Args:
            conference: Conference name
            
        Returns:
            Model info dict or None if no model exists
        """
        query = """
            SELECT * FROM ml_models
            WHERE conference = :conference AND is_active = true
            ORDER BY training_date DESC
            LIMIT 1
        """
        
        result = await self.db_manager.db.fetch_one(
            query, 
            values={"conference": conference}
        )
        
        if result:
            model_info = dict(result)
            # Parse JSON metrics
            if model_info.get('performance_metrics'):
                model_info['performance_metrics'] = json.loads(
                    model_info['performance_metrics']
                )
            return model_info
        
        return None
    
    async def deactivate_old_models(self, conference: str, keep_latest: int = 3):
        """
        Deactivate old models, keeping only the latest N models.
        
        Args:
            conference: Conference name
            keep_latest: Number of latest models to keep active
        """
        # Get all active models sorted by date
        query = """
            SELECT model_id, version, file_path FROM ml_models
            WHERE conference = :conference AND is_active = true
            ORDER BY training_date DESC
        """
        
        models = await self.db_manager.db.fetch_all(
            query,
            values={"conference": conference}
        )
        
        if len(models) <= keep_latest:
            return
        
        # Deactivate older models
        models_to_deactivate = models[keep_latest:]
        
        for model in models_to_deactivate:
            # Deactivate in database
            update_query = """
                UPDATE ml_models 
                SET is_active = false 
                WHERE model_id = :model_id
            """
            await self.db_manager.db.execute(
                update_query,
                values={"model_id": model['model_id']}
            )
            
            # Optionally delete file to save space
            try:
                file_path = Path(model['file_path'])
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted old model file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete model file: {e}")
        
        logger.info(f"Deactivated {len(models_to_deactivate)} old models for {conference}")
    
    async def get_model_performance_history(
        self, 
        conference: str, 
        limit: int = 10
    ) -> List[Dict]:
        """
        Get performance history of models for a conference.
        
        Args:
            conference: Conference name
            limit: Number of records to return
            
        Returns:
            List of model performance records
        """
        query = """
            SELECT 
                version,
                training_date,
                training_games_count,
                performance_metrics,
                is_active
            FROM ml_models
            WHERE conference = :conference
            ORDER BY training_date DESC
            LIMIT :limit
        """
        
        results = await self.db_manager.db.fetch_all(
            query,
            values={"conference": conference, "limit": limit}
        )
        
        # Parse JSON metrics
        history = []
        for row in results:
            record = dict(row)
            if record.get('performance_metrics'):
                record['performance_metrics'] = json.loads(
                    record['performance_metrics']
                )
            history.append(record)
        
        return history
    
    def get_model_path(self, conference: str, version: str = None) -> Path:
        """
        Get the file path for a model.
        
        Args:
            conference: Conference name
            version: Optional version (uses timestamp if not provided)
            
        Returns:
            Path object for model file
        """
        if not version:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"mlsnp_{conference}_{version}.pkl"
        return self.model_base_path / filename
    
    async def cleanup_orphaned_files(self):
        """
        Clean up model files that are not registered in the database.
        """
        # Get all registered file paths
        query = "SELECT file_path FROM ml_models"
        results = await self.db_manager.db.fetch_all(query)
        registered_paths = {Path(row['file_path']) for row in results}
        
        # Check all files in model directory
        orphaned_count = 0
        for file_path in self.model_base_path.glob("mlsnp_*.pkl"):
            if file_path not in registered_paths:
                try:
                    file_path.unlink()
                    orphaned_count += 1
                    logger.info(f"Deleted orphaned model file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete orphaned file {file_path}: {e}")
        
        if orphaned_count > 0:
            logger.info(f"Cleaned up {orphaned_count} orphaned model files")
    
    async def export_model_metadata(self, model_id: int) -> Dict[str, Any]:
        """
        Export model metadata for backup or transfer.
        
        Args:
            model_id: Database model ID
            
        Returns:
            Complete model metadata dict
        """
        query = """
            SELECT * FROM ml_models WHERE model_id = :model_id
        """
        
        result = await self.db_manager.db.fetch_one(
            query,
            values={"model_id": model_id}
        )
        
        if not result:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = dict(result)
        
        # Parse JSON fields
        if metadata.get('performance_metrics'):
            metadata['performance_metrics'] = json.loads(
                metadata['performance_metrics']
            )
        
        # Add export timestamp
        metadata['exported_at'] = datetime.now().isoformat()
        
        return metadata
    
    async def schedule_model_training(
        self,
        conference: str,
        schedule_type: str = "weekly"
    ) -> Dict[str, str]:
        """
        Set up schedule for model retraining (placeholder for cron job setup).
        
        Args:
            conference: Conference to train for
            schedule_type: 'daily' or 'weekly'
            
        Returns:
            Dict with schedule information
        """
        schedules = {
            'daily': '0 2 * * *',     # 2 AM daily
            'weekly': '0 2 * * 0',    # 2 AM Sunday
        }
        
        cron_expression = schedules.get(schedule_type, schedules['weekly'])
        
        # This would be used to set up actual cron job or Railway scheduled job
        schedule_info = {
            'conference': conference,
            'schedule_type': schedule_type,
            'cron_expression': cron_expression,
            'next_run': self._calculate_next_run(cron_expression),
            'command': f'python train_model.py --conference {conference}'
        }
        
        logger.info(f"Model training schedule for {conference}: {schedule_info}")
        
        return schedule_info
    
    def _calculate_next_run(self, cron_expression: str) -> str:
        """Calculate next run time from cron expression (simplified)."""
        # This is a placeholder - in production you'd use a cron parser
        from datetime import timedelta
        
        if '* * 0' in cron_expression:  # Weekly
            days_until_sunday = (6 - datetime.now().weekday()) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7
            next_run = datetime.now() + timedelta(days=days_until_sunday)
        else:  # Daily
            next_run = datetime.now() + timedelta(days=1)
        
        return next_run.replace(hour=2, minute=0, second=0).isoformat()


class PredictorFactory:
    """
    Factory for creating the appropriate predictor based on method.
    """
    
    @staticmethod
    def create_predictor(
        method: str,
        conference: str,
        conference_teams: Dict[str, str],
        games_data: List[Dict],
        team_performance: Dict[str, Dict],
        league_averages: Dict[str, float],
        model_path: Optional[str] = None
    ) -> BasePredictor:
        """
        Create a predictor instance based on the specified method.
        
        Args:
            method: Prediction method ('monte_carlo', 'ml', or 'machine_learning')
            conference: Conference name
            conference_teams: Team mapping
            games_data: Game data
            team_performance: Team performance metrics
            league_averages: League averages
            model_path: Optional path to ML model
            
        Returns:
            BasePredictor instance
            
        Raises:
            ValueError: If method is not recognized
        """
        # Normalize method name
        method = method.lower().replace('_', '').replace('-', '')
        
        if method in ['montecarlo', 'mc', 'traditional', 'poisson']:
            logger.info(f"Creating Monte Carlo predictor for {conference} conference")
            return MonteCarloPredictor(
                conference=conference,
                conference_teams=conference_teams,
                games_data=games_data,
                team_performance=team_performance,
                league_averages=league_averages
            )
        
        elif method in ['ml', 'machinelearning', 'automl', 'ai']:
            logger.info(f"Creating ML predictor for {conference} conference")
            return MLPredictor(
                conference=conference,
                conference_teams=conference_teams,
                games_data=games_data,
                team_performance=team_performance,
                league_averages=league_averages,
                model_path=model_path
            )
        
        else:
            raise ValueError(
                f"Unknown prediction method: {method}. "
                "Valid options: 'monte_carlo', 'ml'"
            )
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available prediction methods."""
        # Check if ML is available
        try:
            if AUTOML_AVAILABLE:
                return ['monte_carlo', 'ml']
            else:
                logger.warning("ML libraries not available")
                return ['monte_carlo']
        except ImportError:
            return ['monte_carlo']
    
    @staticmethod
    def get_method_description(method: str) -> str:
        """Get a description of the prediction method."""
        descriptions = {
            'monte_carlo': (
                "Monte Carlo simulation using Poisson distributions based on xG data. "
                "Runs thousands of season simulations to estimate probabilities."
            ),
            'ml': (
                "Machine Learning prediction using AutoML (AutoGluon/sklearn). "
                "Directly predicts game outcomes based on team features and historical performance."
            )
        }
        
        method = method.lower()
        return descriptions.get(method, f"Unknown method: {method}")
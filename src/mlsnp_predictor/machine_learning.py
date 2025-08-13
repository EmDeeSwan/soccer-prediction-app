from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
import pickle
from pathlib import Path
from scipy import stats
from src.common.utils import time_it
from src.common.BasePredictor import BasePredictor

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

logger = logging.getLogger(__name__)

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
        """
        training_records = []
        
        # Sort games by date to ensure temporal ordering
        # Ensure date parsing is robust
        for g in self.completed_games:
            if isinstance(g.get('date'), str):
                g['date'] = pd.to_datetime(g['date'], errors='coerce')
        
        # Filter out games with invalid dates
        valid_games = [g for g in self.completed_games if pd.notna(g.get('date'))]
        sorted_games = sorted(valid_games, key=lambda x: x['date'])

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
    def run_simulations(self, n_simulations: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict, Dict]:
        """
        Run ML predictions (no simulations needed).
        
        Args:
            n_simulations: Ignored for ML predictions
            
        Returns:
            Tuple of summary DataFrame, empty dict, empty DataFrame, qualification data, and feature importance
        """
        logger.info(f"Running ML predictions for {self.conference} conference")
        
        if not self.ml_model:
            logger.warning("No ML model available. Train a model first or fall back to Monte Carlo.")
            # Ensure fallback returns 5 items
            summary_df, sim_results, rank_df, qual_data = self._run_fallback_predictions()
            return summary_df, sim_results, rank_df, qual_data, {}

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

        # Extract feature importance
        feature_importance = self.get_feature_importance()
        
        # Create summary
        summary_df, qualification_data = self._create_summary(team_projections, playoff_probs)
        
        return summary_df, {}, pd.DataFrame(), qualification_data, feature_importance
    
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
        expected_points = defaultdict(float)
        
        for pred in game_predictions:
            game = pred['game']
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]
            
            # Expected points from win/loss/draw
            home_exp = pred['home_win_prob'] * 3 + pred['draw_prob'] * 1
            away_exp = pred['away_win_prob'] * 3 + pred['draw_prob'] * 1
            
            # Expected points from shootout
            if pred['home_pred'] > pred['away_pred']:
                home_shootout_win_prob = 0.55
            elif pred['away_pred'] > pred['home_pred']:
                home_shootout_win_prob = 0.45
            else:
                home_shootout_win_prob = 0.50
            
            home_exp += pred['draw_prob'] * home_shootout_win_prob
            away_exp += pred['draw_prob'] * (1 - home_shootout_win_prob)

            expected_points[home_id] += home_exp
            expected_points[away_id] += away_exp
            
        # Add current points to get final projection
        final_expected_points = {}
        for team_id in self.conference_teams:
            current_points = self.current_standings.get(team_id, {}).get('points', 0)
            final_expected_points[team_id] = current_points + expected_points.get(team_id, 0)

        return final_expected_points

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
            points_std = np.sqrt(games_remaining) * 1.2 # Std dev of points in a game is ~1.2
            
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from the trained ML model.
        """
        if not self.ml_model:
            logger.warning("No ML model available for feature importance.")
            return {}

        try:
            if AUTOML_LIBRARY == "autogluon":
                training_data = self._prepare_training_data()
                if training_data.empty or len(training_data) < 20:
                    logger.warning("Not enough data to calculate feature importance.")
                    return {}

                logger.info(f"Calculating feature importance using {len(training_data)} training samples.")
                
                # I'm not sure that the subsample_size should exceed the number of rows in the data
                importance_df = self.ml_model.feature_importance(
                    data=training_data,
                    subsample_size=min(1000, len(training_data)),
                    num_shuffle_sets=15,
                    include_confidence_band=True,
                    time_limit=120  # Allow 2 minutes for calculation
                )

                # Log stability info
                logger.info(f"Feature importance calculated with {len(training_data)} samples, "
                            f"20 shuffle sets for improved stability")
                
                # AutoGluon returns a DataFrame. Convert the 'importance' column to a dict.
                importance_dict = importance_df['importance'].to_dict()
                return self._format_feature_names(importance_dict)

            elif AUTOML_LIBRARY == "sklearn":
                if hasattr(self.ml_model.named_steps['rf'], 'feature_importances_'):
                    importances = self.ml_model.named_steps['rf'].feature_importances_
                    feature_importance = dict(zip(self._feature_names, importances))
                    return self._format_feature_names(feature_importance)
                else:
                    return {}
            
            return {}

        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}", exc_info=True)
            return {}
        
    def calculate_feature_importance_variability(self, n_runs: int = 15, 
                                               time_limit_per_run: int = 120) -> Dict[str, Dict]:
        """
        Calculate feature importance variability across multiple model training runs.
        
        Args:
            n_runs: Number of model training runs (default 15)
            time_limit_per_run: Time limit per training run in seconds
            
        Returns:
            Dictionary with feature importance statistics including confidence intervals
        """
        if not AUTOML_AVAILABLE or AUTOML_LIBRARY != "autogluon":
            logger.warning("Feature importance variability requires AutoGluon")
            return {}
        
        logger.info(f"Calculating feature importance variability across {n_runs} model runs...")
        
        # Prepare training data once
        training_data = self._prepare_training_data()
        if len(training_data) < 50:
            logger.warning("Insufficient training data for variability analysis")
            return {}
        
        logger.info(f"Training data prepared: {len(training_data)} samples, {len(training_data.columns)} features")
        
        # Store importance results from each run
        all_importance_results = []
        successful_runs = 0
        failed_runs = []
        
        for run_idx in range(n_runs):
            try:
                logger.info(f"Training model run {run_idx + 1}/{n_runs}...")
                
                # Create unique path for this run
                timestamp = datetime.now().strftime('%H%M%S%f')  # Include microseconds for uniqueness
                model_path = self.model_path.parent / f"variability_run_{run_idx}_{timestamp}"
                
                # Ensure the directory doesn't exist
                if model_path.exists():
                    import shutil
                    shutil.rmtree(model_path)
                
                # Use bootstrap sampling for additional variability
                bootstrap_data = training_data.sample(frac=0.8, replace=True, random_state=run_idx).reset_index(drop=True)
                logger.info(f"   Bootstrap sample: {len(bootstrap_data)} rows")
                
                # Train model for this run with more conservative settings
                try:
                    temp_model = TabularPredictor(
                        label='goals',
                        path=str(model_path),
                        problem_type='regression',
                        eval_metric='root_mean_squared_error',
                        verbosity=1  # Increase verbosity for debugging
                    )
                    
                    # Use faster, more reliable preset
                    temp_model.fit(
                        bootstrap_data,
                        time_limit=max(60, time_limit_per_run // 2),  # Reduce time limit
                        presets='medium_quality',
                        num_bag_folds=2,  # Reduce complexity
                        num_bag_sets=1,
                        num_stack_levels=0,  # Disable stacking for speed
                        verbosity=1
                    )
                    
                    logger.info(f"   Model training completed for run {run_idx + 1}")
                    
                except Exception as e:
                    logger.error(f"   Model training failed for run {run_idx}: {e}")
                    failed_runs.append(f"Run {run_idx}: Model training - {str(e)}")
                    continue
                    
                # Calculate feature importance for this run
                try:
                    # Use smaller subsample and fewer shuffles for reliability
                    subsample_size = min(200, len(bootstrap_data))
                    
                    importance_df = temp_model.feature_importance(
                        data=bootstrap_data,
                        subsample_size=subsample_size,
                        num_shuffle_sets=3,  # Fewer shuffles for reliability
                        time_limit=30  # Shorter time limit
                    )
                    
                    if importance_df is not None and len(importance_df) > 0:
                        # Extract importance scores
                        importance_dict = importance_df['importance'].to_dict()
                        all_importance_results.append(importance_dict)
                        successful_runs += 1
                        logger.info(f"   Feature importance calculated: {len(importance_dict)} features")
                    else:
                        logger.warning(f"   Empty importance results for run {run_idx}")
                        failed_runs.append(f"Run {run_idx}: Empty importance results")
                    
                except Exception as e:
                    logger.error(f"   Failed to calculate importance for run {run_idx}: {e}")
                    failed_runs.append(f"Run {run_idx}: Feature importance - {str(e)}")
                
                # Clean up model files to save space
                try:
                    import shutil
                    if model_path.exists():
                        shutil.rmtree(model_path)
                        logger.debug(f"   Cleaned up model files for run {run_idx}")
                except Exception as e:
                    logger.warning(f"   Failed to clean up model files: {e}")
                    
            except Exception as e:
                logger.error(f"   Unexpected error in run {run_idx}: {e}")
                failed_runs.append(f"Run {run_idx}: Unexpected error - {str(e)}")
                continue
        
        # Log summary of results
        logger.info(f"Variability analysis completed: {successful_runs}/{n_runs} successful runs")
        if failed_runs:
            logger.warning(f"Failed runs details:")
            for failure in failed_runs[:5]:  # Show first 5 failures
                logger.warning(f"   {failure}")
            if len(failed_runs) > 5:
                logger.warning(f"   ... and {len(failed_runs) - 5} more failures")
        
        if successful_runs < 3:
            logger.error(f"Only {successful_runs} successful runs, need at least 3 for variability analysis")
            # Return basic feature importance instead
            logger.info("Falling back to single model feature importance")
            basic_importance = self.get_feature_importance()
            if basic_importance:
                # Create fake variability stats from single model
                fake_stats = {}
                for feature, importance in basic_importance.items():
                    fake_stats[feature] = {
                        'mean': importance,
                        'std': 0.0,
                        'ci_lower': importance,
                        'ci_upper': importance,
                        'coefficient_of_variation': 0.0,
                        'n_runs': 1,
                        'stability_score': 100.0
                    }
                return fake_stats
            return {}
        
        logger.info(f"Successfully completed {successful_runs}/{n_runs} model runs")
        
        # Calculate variability statistics
        return self._calculate_importance_statistics(all_importance_results)

    # Also add this helper method for debugging:
    def get_feature_importance_debug(self) -> Dict[str, float]:
        """Get feature importance with detailed debugging info."""
        if not self.ml_model:
            logger.warning("No ML model available for feature importance.")
            return {}

        try:
            logger.info("Calculating feature importance with debugging...")
            
            if AUTOML_LIBRARY == "autogluon":
                training_data = self._prepare_training_data()
                if training_data.empty or len(training_data) < 20:
                    logger.warning("Not enough data to calculate feature importance.")
                    return {}

                logger.info(f"Using {len(training_data)} training samples for feature importance")
                
                # Use conservative settings
                importance_df = self.ml_model.feature_importance(
                    data=training_data,
                    subsample_size=min(500, len(training_data)),
                    num_shuffle_sets=5,
                    time_limit=60
                )

                if importance_df is not None and len(importance_df) > 0:
                    importance_dict = importance_df['importance'].to_dict()
                    logger.info(f"Successfully calculated importance for {len(importance_dict)} features")
                    return self._format_feature_names(importance_dict)
                else:
                    logger.warning("Feature importance calculation returned empty results")
                    return {}

            elif AUTOML_LIBRARY == "sklearn":
                if hasattr(self.ml_model.named_steps['rf'], 'feature_importances_'):
                    importances = self.ml_model.named_steps['rf'].feature_importances_
                    feature_importance = dict(zip(self._feature_names, importances))
                    return self._format_feature_names(feature_importance)
                else:
                    return {}
            
            return {}

        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}", exc_info=True)
            return {}
    
    def _calculate_importance_statistics(self, all_importance_results: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate statistics from multiple feature importance runs.
        """
        # Get all features that appeared in any run
        all_features = set()
        for result in all_importance_results:
            all_features.update(result.keys())
        
        importance_stats = {}
        
        for feature in all_features:
            # Collect importance scores for this feature across all runs
            feature_scores = []
            for result in all_importance_results:
                # Use 0 if feature didn't appear in this run
                score = result.get(feature, 0.0)
                feature_scores.append(score)
            
            feature_scores = np.array(feature_scores)
            
            # Calculate statistics
            mean_importance = np.mean(feature_scores)
            std_importance = np.std(feature_scores)
            
            # Calculate 95% confidence interval
            confidence_level = 0.95
            alpha = 1 - confidence_level
            n = len(feature_scores)
            
            if n > 1:
                t_critical = stats.t.ppf(1 - alpha/2, n-1)
                margin_error = t_critical * (std_importance / np.sqrt(n))
                ci_lower = mean_importance - margin_error
                ci_upper = mean_importance + margin_error
            else:
                ci_lower = ci_upper = mean_importance
            
            # Calculate coefficient of variation (relative variability)
            cv = (std_importance / mean_importance * 100) if mean_importance != 0 else 0
            
            importance_stats[feature] = {
                'mean': float(mean_importance),
                'std': float(std_importance),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'coefficient_of_variation': float(cv),
                'n_runs': int(n),
                'stability_score': float(100 - min(cv, 100))  # Higher = more stable
            }
        
        # Sort by mean importance
        sorted_features = sorted(
            importance_stats.items(), 
            key=lambda x: x[1]['mean'], 
            reverse=True
        )
        
        return dict(sorted_features)
    
    def get_feature_importance_with_variability(self, n_runs: int = 15) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """
        Get both point estimates and variability measures for feature importance.
        
        Returns:
            Tuple of (standard_importance, variability_stats)
        """
        # Get standard feature importance
        standard_importance = self.get_feature_importance()
        
        # Get variability statistics
        variability_stats = self.calculate_feature_importance_variability(n_runs)
        
        return standard_importance, variability_stats
        
    def _format_feature_names(self, feature_importance: Dict) -> Dict[str, float]:
        """Convert technical feature names to readable ones."""
        name_mapping = {
            'is_home': 'Home Field Advantage',
            'team_xgf_per_game': 'Team Attack Strength (xG)',
            'team_xga_per_game': 'Team Defense Strength (xG)',
            'opp_xgf_per_game': 'Opponent Attack Strength',
            'opp_xga_per_game': 'Opponent Defense Strength', 
            'xg_diff': 'Team Goal Difference (xG)',
            'opp_xg_diff': 'Opponent Goal Difference (xG)',
            'team_form_points': 'Team Recent Form (Points)',
            'team_form_gf': 'Team Recent Goals For',
            'team_form_ga': 'Team Recent Goals Against',
            'opp_form_points': 'Opponent Recent Form',
            'opp_form_gf': 'Opponent Recent Goals For',
            'opp_form_ga': 'Opponent Recent Goals Against',
            'h2h_win_rate': 'Head-to-Head Record',
            'h2h_goals_for_avg': 'H2H Goals For Average',
            'h2h_goals_against_avg': 'H2H Goals Against Average',
            'h2h_games_played': 'H2H Games Played',
            'team_rest_days': 'Team Rest Days',
            'opp_rest_days': 'Opponent Rest Days',
            'month': 'Season Month',
            'day_of_week': 'Day of Week',
            'is_weekend': 'Weekend Game'
        }
        
        formatted = {}
        for feature, importance in feature_importance.items():
            readable_name = name_mapping.get(feature, feature.replace('_', ' ').title())
            # Only include features with positive importance
            if importance > 0:
                formatted[readable_name] = float(importance)
        
        # Sort by importance and return top 15
        sorted_features = sorted(formatted.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_features[:15])
    
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
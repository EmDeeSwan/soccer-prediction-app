import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from scipy import stats

# AutoML imports - AutoGluon for Python 3.12 compatibility
try:
    from autogluon.tabular import TabularPredictor
    AUTOML_AVAILABLE = True
    AUTOML_LIBRARY = "autogluon"
except ImportError:
    try:
        # Fallback to sklearn if no AutoML available
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        AUTOML_AVAILABLE = True
        AUTOML_LIBRARY = "sklearn"
    except ImportError:
        AUTOML_AVAILABLE = False
        AUTOML_LIBRARY = None
        logging.warning("No AutoML library found. Install with: pip install autogluon")

logger = logging.getLogger(__name__)

class MLSNPRegSeasonPredictor:
    """
    DIRECT ML PREDICTION - No Monte Carlo simulations needed!
    Uses ML model to predict all remaining games and calculate expected outcomes.
    """

    def __init__(self, conference: str, conference_teams: Dict[str, str], 
                 games_data: List[Dict], team_performance: Dict[str, Dict], 
                 league_averages: Dict[str, float], use_automl: bool = True):
        """Initialize the predictor with all necessary data."""
        self.conference = conference
        self.conference_teams = set(conference_teams.keys())
        self.team_names = conference_teams
        self.games_data = games_data
        self.team_performance = team_performance
        self.league_avg_xgf = league_averages.get('league_avg_xgf', 1.2)
        self.league_avg_xga = league_averages.get('league_avg_xga', 1.2)
        self.use_automl = use_automl and AUTOML_AVAILABLE

        self.current_standings = self._calculate_current_standings()
        self.remaining_games = self._filter_remaining_games()
        
        # Debug logging
        logger.info(f"=== GAME FILTERING DEBUG ===")
        logger.info(f"Total games passed to predictor: {len(self.games_data)}")
        logger.info(f"Conference teams: {len(self.conference_teams)}")
        logger.info(f"Remaining games after filtering: {len(self.remaining_games)}")
        
        # Pre-compute features for efficiency
        self._precompute_features()
        
        # Initialize ML model
        self.ml_model = None
        self.model_path = Path(f"models/xg_predictor_{conference}_{datetime.now().strftime('%Y%m')}.pkl")
        
        if self.use_automl:
            self._initialize_ml_model()

    def _precompute_features(self):
        """Pre-compute expensive features to avoid recalculating."""
        logger.info("Pre-computing features for ML optimization...")
        
        self.completed_games = [g for g in self.games_data if g.get('is_completed')]
        self.team_form_cache = {}
        self.team_h2h_cache = {}
        self.team_rest_cache = {}
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for team_id in self.conference_teams:
            self.team_form_cache[team_id] = self._calculate_form_optimized(team_id, current_date)
            self.team_rest_cache[team_id] = self._calculate_rest_days_optimized(team_id, current_date)
            
            self.team_h2h_cache[team_id] = {}
            for opponent_id in self.conference_teams:
                if opponent_id != team_id:
                    self.team_h2h_cache[team_id][opponent_id] = self._calculate_h2h_optimized(team_id, opponent_id)
        
        logger.info(f"✅ Pre-computed features for {len(self.conference_teams)} teams")

    def _calculate_form_optimized(self, team_id: str, reference_date: str, n_games: int = 5) -> Dict[str, float]:
        """Optimized form calculation"""
        recent_games = []
        
        for game in reversed(self.completed_games):
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                recent_games.append(game)
            if int(len(recent_games)) >= 5:
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

    def _calculate_h2h_optimized(self, team_id: str, opponent_id: str) -> Dict[str, float]:
        """Optimized H2H calculation"""
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

    def _calculate_rest_days_optimized(self, team_id: str, reference_date: str) -> float:
        """Optimized rest days calculation"""
        try:
            current_date = pd.to_datetime(reference_date)
        except:
            return 7.0
        
        for game in reversed(self.completed_games):
            if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                try:
                    gd = pd.to_datetime(game.get('date'))
                    if gd < current_date:
                        return min((current_date - gd).days, 14)
                except:
                    continue
        
        return 7.0

    def _initialize_ml_model(self):
        """Initialize or load the AutoML model"""
        if self.model_path.exists():
            try:
                if AUTOML_LIBRARY == "autogluon":
                    self.ml_model = TabularPredictor.load(str(self.model_path))
                else:
                    with open(self.model_path, 'rb') as f:
                        self.ml_model = pickle.load(f)
                logger.info(f"Loaded existing ML model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
        
        logger.info(f"Building new AutoML model using {AUTOML_LIBRARY}...")
        training_data = self._prepare_training_data()
        
        if len(training_data) < 50:
            logger.warning("Insufficient training data for ML model. Falling back to traditional method.")
            self.use_automl = False
            return
        
        try:
            if AUTOML_LIBRARY == "autogluon":
                self.model_path.parent.mkdir(exist_ok=True)
                
                self.ml_model = TabularPredictor(
                    label='goals',
                    path=str(self.model_path),
                    problem_type='regression',
                    eval_metric='root_mean_squared_error'
                )
                
                self.ml_model.fit(
                    training_data,
                    time_limit=300,
                    presets='best_quality',
                    verbosity=0
                )
                
                logger.info("AutoGluon model training completed")
                
            elif AUTOML_LIBRARY == "sklearn":
                features = [col for col in training_data.columns if col != 'goals']
                X = training_data[features]
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
                
                self.model_path.parent.mkdir(exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.ml_model, f)
                
                self._feature_names = features
                logger.info("Sklearn RandomForest model training completed")
                
        except Exception as e:
            logger.error(f"AutoML model training failed: {e}")
            self.use_automl = False

    def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data"""
        training_records = []
        
        for game in self.completed_games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue
            
            home_features = self._extract_features_fast(home_id, away_id, True)
            home_features['goals'] = game.get('home_score', 0)
            training_records.append(home_features)
            
            away_features = self._extract_features_fast(away_id, home_id, False)
            away_features['goals'] = game.get('away_score', 0)
            training_records.append(away_features)
        
        return pd.DataFrame(training_records)
    
    def _extract_features(self, team_id: str, opponent_id: str, is_home: bool, 
                         game_date: str, games_before: List[Dict]) -> Dict[str, float]:
        """
        Extract features for ML model prediction.
        
        This is where the magic happens - good feature engineering is crucial!
        """
        features = {}
        
        # Basic features
        features['is_home'] = 1.0 if is_home else 0.0
        
        # Team strength features (from xG data)
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
        
        # Form features (last 5 games)
        team_form = self._calculate_form_optimized(team_id, game_date, games_before)
        opp_form = self._calculate_form_optimized(opponent_id, game_date, games_before)
        
        features['team_form_points'] = team_form['points_per_game']
        features['team_form_gf'] = team_form['goals_for_per_game']
        features['team_form_ga'] = team_form['goals_against_per_game']
        features['opp_form_points'] = opp_form['points_per_game']
        features['opp_form_gf'] = opp_form['goals_for_per_game']
        features['opp_form_ga'] = opp_form['goals_against_per_game']
        
        # Head-to-head features
        h2h = self._calculate_h2h_features(team_id, opponent_id, game_date, games_before)
        features.update(h2h)
        
        # Rest days (if we can calculate it)
        features['team_rest_days'] = self._calculate_rest_days(team_id, game_date, games_before)
        features['opp_rest_days'] = self._calculate_rest_days(opponent_id, game_date, games_before)
        
        # Temporal features
        if game_date:
            try:
                dt = pd.to_datetime(game_date)
                features['month'] = dt.month
                features['day_of_week'] = dt.dayofweek
                features['is_weekend'] = 1.0 if dt.dayofweek >= 5 else 0.0
            except:
                features['month'] = 6  # Default to June
                features['day_of_week'] = 3  # Default to Wednesday
                features['is_weekend'] = 0.0
        
        # League average features for context
        features['league_avg_xgf'] = self.league_avg_xgf
        features['league_avg_xga'] = self.league_avg_xga
        
        return features

    def _extract_features_fast(self, team_id: str, opponent_id: str, is_home: bool) -> Dict[str, float]:
        """Fast feature extraction using pre-computed values"""
        features = {}
        
        features['is_home'] = 1.0 if is_home else 0.0
        
        # Team strength features
        team_stats = self.team_performance.get(team_id, {})
        opp_stats = self.team_performance.get(opponent_id, {})
        
        team_games = max(team_stats.get('games_played', 1), 1)
        opp_games = max(opp_stats.get('games_played', 1), 1)
        
        features['team_xgf_per_game'] = team_stats.get('x_goals_for', 0) / team_games
        features['team_xga_per_game'] = team_stats.get('x_goals_against', 0) / team_games
        features['opp_xgf_per_game'] = opp_stats.get('x_goals_for', 0) / opp_games
        features['opp_xga_per_game'] = opp_stats.get('x_goals_against', 0) / opp_games
        
        features['xg_diff'] = features['team_xgf_per_game'] - features['team_xga_per_game']
        features['opp_xg_diff'] = features['opp_xgf_per_game'] - features['opp_xga_per_game']
        
        # Pre-computed form features
        team_form = self.team_form_cache.get(team_id, {
            'points_per_game': 1.0, 'goals_for_per_game': self.league_avg_xgf, 'goals_against_per_game': self.league_avg_xga
        })
        opp_form = self.team_form_cache.get(opponent_id, {
            'points_per_game': 1.0, 'goals_for_per_game': self.league_avg_xgf, 'goals_against_per_game': self.league_avg_xga
        })
        
        features.update({
            'team_form_points': team_form['points_per_game'],
            'team_form_gf': team_form['goals_for_per_game'],
            'team_form_ga': team_form['goals_against_per_game'],
            'opp_form_points': opp_form['points_per_game'],
            'opp_form_gf': opp_form['goals_for_per_game'],
            'opp_form_ga': opp_form['goals_against_per_game']
        })
        
        # Pre-computed H2H features
        h2h = self.team_h2h_cache.get(team_id, {}).get(opponent_id, {
            'h2h_games_played': 0.0, 'h2h_win_rate': 0.5,
            'h2h_goals_for_avg': self.league_avg_xgf, 'h2h_goals_against_avg': self.league_avg_xga
        })
        features.update(h2h)
        
        # Pre-computed rest days
        features['team_rest_days'] = self.team_rest_cache.get(team_id, 7.0)
        features['opp_rest_days'] = self.team_rest_cache.get(opponent_id, 7.0)
        
        # Temporal features
        current_time = datetime.now()
        features.update({
            'month': current_time.month,
            'day_of_week': current_time.weekday(),
            'is_weekend': 1.0 if current_time.weekday() >= 5 else 0.0,
            'league_avg_xgf': self.league_avg_xgf,
            'league_avg_xga': self.league_avg_xga
        })
        
        return features

    def _calculate_current_standings(self) -> Dict[str, Dict]:
        """Calculate current standings"""
        standings = defaultdict(lambda: {
            "team_id": None, "name": "", "points": 0, "goal_difference": 0,
            "games_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "shootout_wins": 0
        })

        completed_games = [g for g in self.games_data if g.get("is_completed")]
        
        for game in completed_games:
            home_id, away_id = game["home_team_id"], game["away_team_id"]
            if home_id not in self.conference_teams or away_id not in self.conference_teams:
                continue

            for team_id in [home_id, away_id]:
                if standings[team_id]["team_id"] is None:
                    standings[team_id]["team_id"] = team_id
                    standings[team_id]["name"] = self.team_names.get(team_id, f"Team {team_id}")

            home_score, away_score = game.get("home_score", 0), game.get("away_score", 0)

            if game.get("went_to_shootout"):
                home_pens, away_pens = game.get("home_penalties", 0), game.get("away_penalties", 0)
                if home_pens > away_pens:
                    self._update_shootout_winner(standings[home_id], home_score, away_score)
                    self._update_shootout_loser(standings[away_id], away_score, home_score)
                else:
                    self._update_shootout_winner(standings[away_id], away_score, home_score)
                    self._update_shootout_loser(standings[home_id], home_score, away_score)
            else:
                if home_score > away_score:
                    self._update_team_standings(standings[home_id], home_score, away_score, "win")
                    self._update_team_standings(standings[away_id], away_score, home_score, "loss")
                elif away_score > home_score:
                    self._update_team_standings(standings[away_id], away_score, home_score, "win")
                    self._update_team_standings(standings[home_id], home_score, away_score, "loss")

        for team_id, stats in standings.items():
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]

        return {team_id: dict(stats) for team_id, stats in standings.items()}

    def _filter_remaining_games(self) -> List[Dict]:
        """Filter for future games"""
        return [
            game for game in self.games_data
            if not game.get("is_completed") and
               game.get("home_team_id") in self.conference_teams and
               game.get("away_team_id") in self.conference_teams
        ]

    def run_simulations(self, n_simulations: int = None) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        DIRECT ML PREDICTION - No simulations needed!
        Predicts all remaining games using ML model and calculates expected outcomes.
        """
        logger.info("🎯 Using DIRECT ML PREDICTION (no Monte Carlo simulations)")
        logger.info(f"Predicting outcomes for {len(self.remaining_games)} remaining games...")
        
        if not self.use_automl or not self.ml_model:
            logger.warning("ML model not available, falling back to traditional method")
            return self._run_traditional_predictions()
        
        # Step 1: Predict all remaining games
        game_predictions = []
        
        for i, game in enumerate(self.remaining_games):
            if i % 20 == 0:
                logger.info(f"   Predicting game {i+1}/{len(self.remaining_games)}...")
            
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]
            
            # Get ML predictions for both teams
            home_features = self._extract_features_fast(home_id, away_id, True)
            away_features = self._extract_features_fast(away_id, home_id, False)
            
            if AUTOML_LIBRARY == "autogluon":
                home_pred = self.ml_model.predict(pd.DataFrame([home_features]))[0]
                away_pred = self.ml_model.predict(pd.DataFrame([away_features]))[0]
            else:  # sklearn
                home_clean = {k: v for k, v in home_features.items() 
                             if k not in ['goals', 'league_avg_xgf', 'league_avg_xga']}
                away_clean = {k: v for k, v in away_features.items() 
                             if k not in ['goals', 'league_avg_xgf', 'league_avg_xga']}
                
                home_X = pd.DataFrame([home_clean])[self._feature_names]
                away_X = pd.DataFrame([away_clean])[self._feature_names]
                
                home_pred = self.ml_model.predict(home_X)[0]
                away_pred = self.ml_model.predict(away_X)[0]
            
            # Ensure positive predictions
            home_pred = max(0.1, home_pred)
            away_pred = max(0.1, away_pred)
            
            # Calculate win probabilities using Poisson distributions
            home_win_prob = self._calculate_win_probability(home_pred, away_pred, True)
            away_win_prob = self._calculate_win_probability(away_pred, home_pred, True)
            draw_prob = 1.0 - home_win_prob - away_win_prob
            
            game_predictions.append({
                'game': game,
                'home_pred': home_pred,
                'away_pred': away_pred,
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob,
                'draw_prob': draw_prob
            })
        
        logger.info("✅ Game predictions complete! Calculating expected points...")
        
        # Step 2: Calculate expected points for each team
        expected_points = {}
        for team_id in self.conference_teams:
            current_points = self.current_standings.get(team_id, {}).get('points', 0)
            expected_additional_points = 0
            
            for pred in game_predictions:
                game = pred['game']
                if game["home_team_id"] == team_id:
                    # Team is home
                    expected_additional_points += (
                        pred['home_win_prob'] * 3 +  # 3 pts for win
                        pred['draw_prob'] * 1.5       # avg 1.5 pts for draw (shootout)
                    )
                elif game["away_team_id"] == team_id:
                    # Team is away
                    expected_additional_points += (
                        pred['away_win_prob'] * 3 +   # 3 pts for win
                        pred['draw_prob'] * 1.5       # avg 1.5 pts for draw (shootout)
                    )
            
            expected_points[team_id] = current_points + expected_additional_points
        
        # Step 3: Calculate playoff probabilities using normal distribution
        logger.info("📊 Calculating playoff probabilities...")
        
        # Sort teams by expected points
        team_projections = []
        for team_id, exp_points in expected_points.items():
            current_stats = self.current_standings.get(team_id, {})
            
            # Calculate uncertainty (standard deviation) based on remaining games
            games_remaining = sum(1 for pred in game_predictions 
                                if team_id in [pred['game']["home_team_id"], pred['game']["away_team_id"]])
            
            # More games remaining = more uncertainty
            points_std = np.sqrt(games_remaining) * 0.8  # Empirical scaling factor
            
            team_projections.append({
                'team_id': team_id,
                'team_name': self.team_names.get(team_id, team_id),
                'current_points': current_stats.get('points', 0),
                'games_played': current_stats.get('games_played', 0),
                'expected_points': exp_points,
                'points_std': points_std,
                'games_remaining': games_remaining
            })
        
        # Sort by expected points
        team_projections.sort(key=lambda x: x['expected_points'], reverse=True)
        
        # Calculate playoff probabilities using Monte Carlo on final point distributions
        playoff_probs = self._calculate_playoff_probabilities(team_projections)
        
        # Step 4: Create summary DataFrame
        summary_data = []
        for i, proj in enumerate(team_projections):
            team_id = proj['team_id']
            playoff_prob = playoff_probs.get(team_id, 0) * 100
            
            summary_data.append({
                'Team': proj['team_name'],
                '_team_id': team_id,
                'Current Points': proj['current_points'],
                'Games Played': proj['games_played'],
                'Playoff Qualification %': playoff_prob,
                'Average Final Rank': i + 1,  # Based on expected points ranking
                'Average Points': proj['expected_points'],
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values(by='Playoff Qualification %', ascending=False).reset_index(drop=True)
        
        # Create qualification data
        qualification_data = {}
        for proj in team_projections:
            team_id = proj['team_id']
            qualification_data[team_id] = {
                'games_remaining': proj['games_remaining'],
                'status': '',
                'shootout_win_impact': {}
            }
        
        logger.info("✅ Direct ML predictions complete!")
        
        return summary_df, {}, pd.DataFrame(), qualification_data

    def _calculate_win_probability(self, team_goals: float, opp_goals: float, include_shootout: bool = True) -> float:
        """Calculate win probability using Poisson distributions"""
        # Use Poisson to calculate probability of different score outcomes
        max_goals = 8  # Reasonable upper bound for soccer
        win_prob = 0
        
        for home_score in range(max_goals):
            for away_score in range(max_goals):
                home_prob = stats.poisson.pmf(home_score, team_goals)
                away_prob = stats.poisson.pmf(away_score, opp_goals)
                game_prob = home_prob * away_prob
                
                if home_score > away_score:
                    win_prob += game_prob
                elif home_score == away_score and include_shootout:
                    # In shootout, assume 55% chance for the "home" team (or stronger team)
                    shootout_advantage = 0.55 if team_goals >= opp_goals else 0.45
                    win_prob += game_prob * shootout_advantage
        
        return min(win_prob, 0.95)  # Cap at 95% to be realistic

    def _calculate_playoff_probabilities(self, team_projections: List[Dict], n_samples: int = 10000) -> Dict[str, float]:
        """Calculate playoff probabilities using Monte Carlo on point distributions"""
        playoff_counts = defaultdict(int)
        
        for _ in range(n_samples):
            # Sample final points for each team from normal distribution
            sample_points = []
            for proj in team_projections:
                sampled_points = np.random.normal(proj['expected_points'], proj['points_std'])
                sample_points.append((proj['team_id'], max(0, sampled_points)))  # Ensure non-negative
            
            # Sort by sampled points and take top 8
            sample_points.sort(key=lambda x: x[1], reverse=True)
            playoff_teams = [team_id for team_id, _ in sample_points[:8]]
            
            for team_id in playoff_teams:
                playoff_counts[team_id] += 1
        
        # Convert to probabilities
        return {team_id: count / n_samples for team_id, count in playoff_counts.items()}

    def _run_traditional_predictions(self) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """Fallback to traditional xG-based predictions if ML not available"""
        logger.info("Using traditional xG-based predictions...")
        
        expected_points = {}
        for team_id in self.conference_teams:
            current_points = self.current_standings.get(team_id, {}).get('points', 0)
            games_remaining = sum(1 for game in self.remaining_games 
                                if team_id in [game["home_team_id"], game["away_team_id"]])
            
            # Simple estimation: assume team gets 1.5 points per game on average
            team_stats = self.team_performance.get(team_id, {})
            if team_stats.get('games_played', 0) > 0:
                current_ppg = current_points / team_stats['games_played']
            else:
                current_ppg = 1.5  # League average assumption
            
            expected_additional_points = games_remaining * current_ppg
            expected_points[team_id] = current_points + expected_additional_points
        
        # Create summary
        summary_data = []
        sorted_teams = sorted(expected_points.items(), key=lambda x: x[1], reverse=True)
        
        for i, (team_id, exp_points) in enumerate(sorted_teams):
            current_stats = self.current_standings.get(team_id, {})
            playoff_prob = max(0, 100 - (i * 12.5))  # Simple linear decline
            
            summary_data.append({
                'Team': self.team_names.get(team_id, team_id),
                '_team_id': team_id,
                'Current Points': current_stats.get('points', 0),
                'Games Played': current_stats.get('games_played', 0),
                'Playoff Qualification %': playoff_prob,
                'Average Final Rank': i + 1,
                'Average Points': exp_points,
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df, {}, pd.DataFrame(), {}

    # Helper methods (keeping the same as before)
    def _update_team_standings(self, team_stats: Dict, goals_for: int, goals_against: int, result: str):
        team_stats["games_played"] += 1
        team_stats["goals_for"] += goals_for
        team_stats["goals_against"] += goals_against
        team_stats["goal_difference"] += (goals_for - goals_against)
        if result == "win":
            team_stats["wins"] += 1
            team_stats["points"] += 3
        elif result == "loss":
            team_stats["losses"] += 1

    def _update_shootout_winner(self, team_stats: Dict, goals_for: int, goals_against: int):
        self._update_team_standings(team_stats, goals_for, goals_against, "loss")
        team_stats["draws"] += 1
        team_stats["shootout_wins"] += 1
        team_stats["points"] += 2

    def _update_shootout_loser(self, team_stats: Dict, goals_for: int, goals_against: int):
        self._update_team_standings(team_stats, goals_for, goals_against, "loss")
        team_stats["draws"] += 1
        team_stats["points"] += 1

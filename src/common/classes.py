from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, ClassVar
import logging
from datetime import datetime
from pathlib import Path
import json
from src.common.database_manager import DatabaseManager
from src.common.BasePredictor import BasePredictor

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

class MatchPredictionRequest(BaseModel):
    team1_id: str
    team2_id: str
    model_type: str = Field("monte_carlo", pattern="^(monte_carlo|ml|both)$")
    n_simulations: Optional[int] = 10000
    
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

        # Delayed imports to prevent circular dependencies
        from src.mlsnp_predictor.monte_carlo import MonteCarloPredictor
        
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
            from src.mlsnp_predictor.machine_learning import MLPredictor
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
        from src.mlsnp_predictor.machine_learning import AUTOML_AVAILABLE
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
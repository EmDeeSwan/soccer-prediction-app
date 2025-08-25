import pandas as pd
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.common.database import database
from src.common.database_manager import DatabaseManager
from src.common.classes import PredictorFactory, MLModelManager
from src.common.utils import Timer

"""
Script to train ML models for MLS Next Pro predictions.
Can be run manually or scheduled (e.g., weekly via cron/Railway).
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/train_model.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)


async def evaluate_model(predictor, test_games):
    """
    Evaluate model performance on test games.
    
    Args:
        predictor: ML predictor instance
        test_games: List of completed games to test on
        
    Returns:
        Dict of performance metrics
    """
    if not test_games:
        return {"error": "No test games available"}
    
    predictions = []
    actuals = []
    failed_predictions = 0

    print(f"Evaluating on {len(test_games)} test games...")
    
    for i, game in enumerate(test_games):
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        
        try:
            # Get features
            home_features = predictor._extract_features(home_id, away_id, True)
            away_features = predictor._extract_features(away_id, home_id, False)
            
            # Debug: Check if features are valid
            if not home_features or not away_features:
                print(f"Game {i}: Empty features - Home: {len(home_features)}, Away: {len(away_features)}")
                failed_predictions += 1
                continue
            
            # Make predictions based on model type
            if hasattr(predictor, '_feature_names') and predictor._feature_names:
                # sklearn model
                print(f"Game {i}: Using sklearn with {len(predictor._feature_names)} features")
                
                # Filter features to match training
                home_clean = {k: v for k, v in home_features.items() if k in predictor._feature_names}
                away_clean = {k: v for k, v in away_features.items() if k in predictor._feature_names}
                
                if len(home_clean) != len(predictor._feature_names):
                    print(f"Game {i}: Feature mismatch - Expected {len(predictor._feature_names)}, got {len(home_clean)}")
                    failed_predictions += 1
                    continue
                
                home_X = pd.DataFrame([home_clean])[predictor._feature_names]
                away_X = pd.DataFrame([away_clean])[predictor._feature_names]
                
                home_pred = predictor.ml_model.predict(home_X)[0]
                away_pred = predictor.ml_model.predict(away_X)[0]
                
            else:
                # AutoGluon model
                if i == 0:  # Only print once
                    print("Using AutoGluon prediction")
                
                # Create DataFrames for AutoGluon
                home_df = pd.DataFrame([home_features])
                away_df = pd.DataFrame([away_features])
                
                # Remove target column if present
                if 'goals' in home_df.columns:
                    home_df = home_df.drop(['goals'], axis=1)
                if 'goals' in away_df.columns:
                    away_df = away_df.drop(['goals'], axis=1)
                
                home_pred = predictor.ml_model.predict(home_df)[0]
                away_pred = predictor.ml_model.predict(away_df)[0]
            
            # Debug first few predictions
            if i < 3:
                print(f"Game {i}: Predictions - Home: {home_pred:.2f}, Away: {away_pred:.2f}")
            
            # Store predictions and actuals
            predictions.extend([home_pred, away_pred])
            actuals.extend([game.get('home_score', 0), game.get('away_score', 0)])
            
        except Exception as e:
            failed_predictions += 1
            if i < 5:  # Only show first few errors
                print(f"Game {i}: Prediction failed - {type(e).__name__}: {e}")
            
            # Check if it's a specific type of error
            if "feature" in str(e).lower():
                print(f"   Feature-related error. Available features: {list(home_features.keys())[:5]}...")
            elif "model" in str(e).lower():
                print(f"   Model-related error. Model type: {type(predictor.ml_model)}")
    
    print(f"Prediction summary: {len(predictions)//2} successful, {failed_predictions} failed")
    
    if not predictions:
        return {"error": "No predictions could be made", "failed_count": failed_predictions}
    
    # Calculate metrics
    import numpy as np
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Calculate match result accuracy
    correct_results = 0
    total_games = len(predictions) // 2
    
    for i in range(total_games):
        if i * 2 + 1 < len(predictions):
            pred_home = predictions[i * 2]
            pred_away = predictions[i * 2 + 1]
            actual_home = actuals[i * 2]
            actual_away = actuals[i * 2 + 1]
            
            # Check if prediction got the result right
            pred_result = "home" if pred_home > pred_away else "away" if pred_away > pred_home else "draw"
            actual_result = "home" if actual_home > actual_away else "away" if actual_away > actual_home else "draw"
            
            if pred_result == actual_result:
                correct_results += 1
    
    result_accuracy = correct_results / total_games if total_games > 0 else 0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "result_accuracy": float(result_accuracy),
        "n_test_games": total_games,
        "n_predictions": len(predictions),
        "failed_predictions": failed_predictions
    }


async def train_model_for_conference(
    conference: str,
    season_year: int,
    db_manager: DatabaseManager,
    model_manager: MLModelManager,
    time_limit: int = 300
):
    """
    Train ML model for a specific conference.
    
    Args:
        conference: Conference name ('eastern' or 'western')
        season_year: Season year
        db_manager: Database manager instance
        model_manager: Model manager instance
        time_limit: Time limit for training in seconds
    """
    print(f"\n{'='*60}")
    print(f"Training ML Model for {conference.upper()} Conference")
    print(f"{'='*60}")
    
    # Get conference data
    conf_id = 1 if conference == 'eastern' else 2
    conference_teams = await db_manager.get_conference_teams(conf_id, season_year)
    
    if not conference_teams:
        logger.error(f"No teams found for {conference} conference")
        return
    
    print(f"Found {len(conference_teams)} teams")
    
    # Get all games
    all_games = await db_manager.get_games_for_season(
        season_year, conference, include_incomplete=True
    )
    
    completed_games = [g for g in all_games if g.get('is_completed')]
    print(f"Total completed games: {len(completed_games)}")
    
    if len(completed_games) < 50:
        logger.error(f"Insufficient games for training ({len(completed_games)} < 50)")
        return
    
    # Split into train and test sets (80/20 split by date)
    games_sorted = sorted(completed_games, key=lambda x: x.get('date', ''))
    split_idx = int(len(games_sorted) * 0.8)
    
    train_games = games_sorted[:split_idx]
    test_games = games_sorted[split_idx:]
    
    print(f"Training set: {len(train_games)} games")
    print(f"Test set: {len(test_games)} games")
    
    # Get team performance data
    team_performance = {}
    for team_id in conference_teams.keys():
        xg_data = await db_manager.get_or_fetch_team_xg(team_id, season_year)
        if xg_data and xg_data.get('games_played', 0) > 0:
            team_performance[team_id] = xg_data
        else:
            team_performance[team_id] = {
                'team_id': team_id,
                'games_played': 1,
                'x_goals_for': 1.2,
                'x_goals_against': 1.2
            }
    
    # Calculate league averages
    from src.mlsnp_predictor.run_predictor import calculate_league_averages
    league_averages = await calculate_league_averages(db_manager, season_year)
    
    # Create predictor
    print(f"\nCreating ML predictor...")
    predictor = PredictorFactory.create_predictor(
        method='ml',
        conference=conference,
        conference_teams=conference_teams,
        games_data=train_games,  # Use only training games
        team_performance=team_performance,
        league_averages=league_averages
    )
    
    # Train model
    print(f"Training model (time limit: {time_limit} seconds)...")
    with Timer() as t:
        success = predictor.train_model(time_limit=time_limit)
    
    if not success:
        logger.error("Model training failed")
        return
    
    training_time = t.elapsed
    print(f"✅ Model trained in {training_time:.1f} seconds")
    
    # Evaluate on test set
    print(f"\nEvaluating model on test set...")
    metrics = await evaluate_model(predictor, test_games)
    mae = metrics.get('mae', 'N/A')
    rmse = metrics.get('rmse', 'N/A')
    accuracy = metrics.get('result_accuracy', 0)
    
    print(f"\n📊 Model Performance:")
    if isinstance(mae, str):
        print(f"   MAE: {mae}")
        print(f"   RMSE: {rmse}")
        print(f"   Result Accuracy: N/A")
    else:
        print(f"   MAE: {mae:.3f} goals")
        print(f"   RMSE: {rmse:.3f} goals")  
        print(f"   Result Accuracy: {accuracy*100:.1f}%")
    print(f"   Test Games: {metrics.get('n_test_games', 0)}")
    
    # Register model in database
    model_path = predictor.model_path
    model_version = predictor.model_version or f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Add training details to metrics
    metrics['training_time'] = training_time
    metrics['training_games'] = len(train_games)
    metrics['test_games'] = len(test_games)
    
    model_id = await model_manager.register_model(
        model_name=f"MLSNP {conference.title()} AutoML",
        conference=conference,
        version=model_version,
        file_path=str(model_path),
        training_games_count=len(train_games),
        performance_metrics=metrics
    )
    
    print(f"\n✅ Model registered with ID: {model_id}")
    
    # Deactivate old models (keep last 3)
    await model_manager.deactivate_old_models(conference, keep_latest=3)
    
    # Show model history
    print(f"\n📈 Recent Model Performance History:")
    history = await model_manager.get_model_performance_history(conference, limit=5)
    
    print(f"{'Version':<20} {'Date':<20} {'MAE':<8} {'RMSE':<8} {'Accuracy':<10}")
    print("-" * 70)
    
    for record in history:
        metrics = record.get('performance_metrics', {})
        print(f"{record['version'][:19]:<20} "
              f"{record['training_date'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{metrics.get('mae', 0):<8.3f} "
              f"{metrics.get('rmse', 0):<8.3f} "
              f"{metrics.get('result_accuracy', 0)*100:<9.1f}%")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train ML models for MLS Next Pro predictions'
    )
    
    parser.add_argument(
        '--conference',
        choices=['eastern', 'western', 'both'],
        default='both',
        help='Conference to train for'
    )
    
    parser.add_argument(
        '--season',
        type=int,
        default=datetime.now().year,
        help='Season year (default: current year)'
    )
    
    parser.add_argument(
        '--time-limit',
        type=int,
        default=300,
        help='Time limit for training in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--update-games',
        action='store_true',
        help='Update incomplete games before training'
    )
    
    args = parser.parse_args()
    
    print(f"\n🤖 MLS Next Pro ML Model Training")
    print(f"   Season: {args.season}")
    print(f"   Conference: {args.conference}")
    print(f"   Time Limit: {args.time_limit} seconds")
    
    try:
        # Connect to database
        print("\n🔌 Connecting to database...")
        await database.connect()
        db_manager = DatabaseManager(database)
        await db_manager.initialize()
        print("✅ Connected!")
        
        # Initialize model manager
        model_manager = MLModelManager(db_manager)
        
        # Update games if requested
        if args.update_games:
            print("\n🔄 Updating game data...")
            if args.conference == 'both':
                await db_manager.update_games_with_asa(args.season, 'eastern')
                await db_manager.update_games_with_asa(args.season, 'western')
            else:
                await db_manager.update_games_with_asa(args.season, args.conference)
            print("✅ Games updated!")
        
        # Determine conferences to train
        if args.conference == 'both':
            conferences = ['eastern', 'western']
        else:
            conferences = [args.conference]
        
        # Train models
        for conference in conferences:
            await train_model_for_conference(
                conference=conference,
                season_year=args.season,
                db_manager=db_manager,
                model_manager=model_manager,
                time_limit=args.time_limit
            )
        
        # Cleanup orphaned files
        print("\n🧹 Cleaning up orphaned model files...")
        await model_manager.cleanup_orphaned_files()
        
        print("\n✅ Training complete!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    finally:
        print("\n🔌 Disconnecting from database...")
        await database.disconnect()


if __name__ == "__main__":
    # Ensure directories exist
    Path("output").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Run
    asyncio.run(main())
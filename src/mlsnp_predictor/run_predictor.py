import logging
import sys
import os
import io
import asyncio
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict
from src.common.database import database
from src.common.database_manager import DatabaseManager
from src.common.classes import PredictorFactory, MLModelManager
from src.common.chart_generator import MLSNPChartGenerator
from src.common.utils import Timer

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/run_predictor.log', encoding='utf-8')
    ],
    force=True  # Ensure handlers are set up even if already configured
)
logger = logging.getLogger(__name__)

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

async def get_user_choices():
    """Get user input for simulation parameters."""
    print("\n" + "="*60)
    print("MLS NEXT PRO SEASON PREDICTOR")
    print("="*60)
    
    # Season year
    while True:
        try:
            season_year = int(input("\nEnter season year (e.g., 2025): ") or "2025")
            break
        except ValueError:
            print("Invalid year. Please enter a number.")
    
    # Conference
    print("\nWhich conference to simulate?")
    print("  1. Eastern")
    print("  2. Western")
    print("  3. Both")
    
    conf_choice = input("Enter choice (1-3): ").strip()
    conference_map = {"1": "eastern", "2": "western", "3": "both"}
    conference = conference_map.get(conf_choice, "eastern")
    
    # Prediction method
    available_methods = PredictorFactory.get_available_methods()
    
    print("\n Choose prediction method:")
    print("  1. Monte Carlo (Traditional xG-based)")
    if 'ml' in available_methods:
        print("  2. Machine Learning (AutoML)")
        print("  3. Both (Compare methods)")
    
    method_choice = input("Enter choice: ").strip()
    
    if 'ml' not in available_methods and method_choice in ["2", "3"]:
        print(" ML not available. Using Monte Carlo only.")
        prediction_method = "monte_carlo"
    else:
        method_map = {
            "1": "monte_carlo",
            "2": "ml",
            "3": "both"
        }
        prediction_method = method_map.get(method_choice, "monte_carlo")
    
    # Number of simulations
    print("\n Simulation presets:")
    print("  1. Quick (1,000 simulations) - ~5 seconds")
    print("  2. Standard (25,000 simulations) - ~1 minute")
    print("  3. Detailed (50,000 simulations) - ~2 minutes")
    print("  4. Research (100,000 simulations) - ~4 minutes")
    print("  5. Custom")
    
    sim_choice = input("Enter choice (1-5): ").strip()
    
    sim_presets = {
        "1": 1000,
        "2": 25000,
        "3": 50000,
        "4": 100000
    }
    
    if sim_choice == "5":
        while True:
            try:
                n_simulations = int(input("Enter number of simulations: "))
                break
            except ValueError:
                print("Invalid number. Please try again.")
    else:
        n_simulations = sim_presets.get(sim_choice, 25000)
    
    # Store results
    store_results = input("\nStore results in database? (y/n): ").lower().strip() == 'y'
    
    # Generate charts
    generate_charts = input("Generate interactive charts? (y/n): ").lower().strip() == 'y'
    
    # Feature importance variability (only for ML methods)
    calculate_variability = False
    if prediction_method in ["ml", "both"] and generate_charts:
        calc_var = input("Calculate feature importance variability? (takes extra time) (y/n): ").lower().strip()
        calculate_variability = calc_var == 'y'
        
        if calculate_variability:
            try:
                var_runs = int(input("Number of variability runs (10-20 recommended): ") or "15")
                var_runs = max(10, min(var_runs, 25))  # Limit between 10-25
            except ValueError:
                var_runs = 15
        else:
            var_runs = 15
    else:
        var_runs = 15
    
    return {
        'season_year': season_year,
        'conference': conference,
        'prediction_method': prediction_method,
        'n_simulations': n_simulations,
        'store_results': store_results,
        'generate_charts': generate_charts,
        'calculate_variability': calculate_variability,
        'variability_runs': var_runs
    }

async def update_game_data(db_manager: DatabaseManager, season_year: int, conference: str):
    """Update incomplete game data from ASA."""
    print(f"\nChecking for game updates...")
    
    # Get game statistics
    query = """
        SELECT 
            COUNT(*) as total_games,
            SUM(CASE WHEN is_completed THEN 1 ELSE 0 END) as completed_games,
            SUM(CASE WHEN NOT is_completed AND date < NOW() THEN 1 ELSE 0 END) as games_to_update
        FROM games 
        WHERE season_year = :season_year
    """
    
    game_stats = await db_manager.db.fetch_one(
        query,
        values={"season_year": season_year}
    )
    
    total_games = game_stats['total_games'] or 0
    completed_games = game_stats['completed_games'] or 0
    games_to_update_count = game_stats['games_to_update'] or 0

    print(f"   Total games: {total_games}")
    print(f"   Completed: {completed_games}")
    print(f"   Need updates: {games_to_update_count}")
    
    if games_to_update_count > 0:
        print(f"   Updating {games_to_update_count} games from ASA API...")
        if conference == 'both':
            await db_manager.update_games_with_asa(season_year, 'eastern')
            await db_manager.update_games_with_asa(season_year, 'western')
        else:
            await db_manager.update_games_with_asa(season_year, conference)
        print("Games updated!")
    else:
        print("   No games require updates at this time.")

async def calculate_league_averages(db_manager: DatabaseManager, season_year: int) -> Dict[str, float]:
    """
    Calculate league-wide averages for all teams for a given season.
    Added safety guards against division by zero.
    """
    # Get all teams' performance data
    try:
        all_teams_xg = await db_manager.db.fetch_all("""
            SELECT 
                team_id,
                x_goals_for,
                x_goals_against,
                games_played,
                date_captured
            FROM team_xg_history
            WHERE season_year = :season_year
            AND games_played > 0
            ORDER BY team_id, date_captured DESC
        """, values={"season_year": season_year})
    except Exception as e:
        logger.error(f"Error fetching team xG data: {e}")
        return {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}
    
    if not all_teams_xg:
        logger.warning(f"No xG data found for season {season_year}")
        return {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}
    
    # Group by team and take most recent data point
    team_latest = {}
    for row in all_teams_xg:
        team_id = row['team_id']
        if team_id not in team_latest:
            team_latest[team_id] = dict(row)
    
    # Calculate weighted averages with validation
    total_xgf = sum(float(t['x_goals_for'] or 0) for t in team_latest.values())
    total_xga = sum(float(t['x_goals_against'] or 0) for t in team_latest.values())
    total_games = sum(int(t['games_played']) for t in team_latest.values())
    
    # Safety guards
    if total_games > 0 and total_xgf > 0 and total_xga > 0:
        league_avg_xgf = total_xgf / total_games
        league_avg_xga = total_xga / total_games
    else:
        logger.warning(f"Using fallback averages. Games: {total_games}, xGF: {total_xgf}, xGA: {total_xga}")
        league_avg_xgf = 1.2
        league_avg_xga = 1.2
    
    # Ensure reasonable bounds
    league_avg_xgf = max(min(league_avg_xgf, 3.0), 0.5)
    league_avg_xga = max(min(league_avg_xga, 3.0), 0.5)
    
    logger.info(f"League averages: xGF={league_avg_xgf:.3f}/game, xGA={league_avg_xga:.3f}/game ({len(team_latest)} teams, {total_games} total games)")
    
    return {
        "league_avg_xgf": league_avg_xgf,
        "league_avg_xga": league_avg_xga,
        "total_teams": len(team_latest),
        "total_games": total_games
    }

async def run_predictions_for_conference(
    conference: str,
    season_year: int,
    n_simulations: int,
    prediction_method: str,
    db_manager: DatabaseManager,
    model_manager: MLModelManager,
    calculate_variability: bool = False,
    variability_runs: int = 15
) -> Dict:
    """Run predictions for a single conference."""
    print(f"\n{'='*60}")
    print(f"Running {prediction_method} predictions for {conference.upper()} conference")
    print(f"{'='*60}")
    
    # Get conference ID
    conf_id = 1 if conference == 'eastern' else 2
    
    # Get teams
    conference_teams = await db_manager.get_conference_teams(conf_id, season_year)
    if not conference_teams:
        print(f" No teams found for {conference} conference")
        return None
    
    print(f"Found {len(conference_teams)} teams")
    
    # Get all necessary data
    print("Loading game and team data...")
    all_games = await db_manager.get_games_for_season(season_year, conference, include_incomplete=True)
    
    # Get team performance data
    team_performance = {}
    for team_id in conference_teams.keys():
        xg_data = await db_manager.get_or_fetch_team_xg(team_id, season_year)
        if xg_data and xg_data.get('games_played', 0) > 0:
            team_performance[team_id] = xg_data
        else:
            # Fallback data
            team_performance[team_id] = {
                'team_id': team_id,
                'games_played': 1,
                'x_goals_for': 1.2,
                'x_goals_against': 1.2
            }
    
    # Calculate league averages
    league_averages = await calculate_league_averages(db_manager, season_year)
    print(f"League averages: xGF={league_averages['league_avg_xgf']:.2f}, xGA={league_averages['league_avg_xga']:.2f}")
    
    results = {}
    
    # Run predictions based on method
    if prediction_method == "both":
        # Run both methods
        methods = ["monte_carlo", "ml"]
    else:
        methods = [prediction_method]
    
    for method in methods:
        print(f"\n🎯 Running {method} predictions...")
        
        if method == "ml":
            # Check for existing model
            model_info = await model_manager.get_latest_model(conference)
            model_path = None
            
            if model_info:
                model_path = model_info['file_path']
                print(f"   Using existing model: {model_info['version']}")
                print(f"   Trained on: {model_info['training_date']}")
            else:
                print("   No existing model found.")
                train_now = input("   Train new model now? (y/n): ").lower().strip() == 'y'
                
                if not train_now:
                    print("   Skipping ML predictions.")
                    continue
        
        # Create predictor
        predictor = PredictorFactory.create_predictor(
            method=method,
            conference=conference,
            conference_teams=conference_teams,
            games_data=all_games,
            team_performance=team_performance,
            league_averages=league_averages,
            model_path=model_path if method == "ml" else None
        )
        
        # Train ML model if needed
        if method == "ml" and not predictor.ml_model:
            print("   Training new ML model (this may take a few minutes)...")
            success = predictor.train_model(time_limit=300)
            
            if success:
                # Register model in database
                model_path = predictor.model_path
                model_id = await model_manager.register_model(
                    model_name=f"MLSNP {conference.title()} Predictor",
                    conference=conference,
                    version=predictor.model_version,
                    file_path=str(model_path),
                    training_games_count=len([g for g in all_games if g.get('is_completed')]),
                    performance_metrics={"training": "successful"}
                )
                print(f"   Model trained and registered (ID: {model_id})")
            else:
                print("   Model training failed")
                continue
        
        # Run predictions
        with Timer() as t:
            if method == "ml":
                # ML returns 5 values including feature importance
                summary_df, sim_results, _, qual_data, feature_importance = predictor.run_simulations(n_simulations)
            # Calculate variability if requested
                variability_stats = {}
                if calculate_variability:
                    print(f"   Calculating feature importance variability ({variability_runs} runs)...")
                    try:
                        _, variability_stats = predictor.get_feature_importance_with_variability(variability_runs)
                        print(f"   Variability analysis completed!")
                    except Exception as e:
                        logger.error(f"Variability calculation failed: {e}")
                        print(f"   Variability analysis failed: {e}")
            else:
                # Monte Carlo returns 4 values
                summary_df, sim_results, _, qual_data = predictor.run_simulations(n_simulations)
                feature_importance = {}
                variability_stats = {}

        elapsed = t.elapsed        
        print(f"   Completed in {elapsed:.1f} seconds")
        
        results[method] = {
            'summary_df': summary_df,
            'simulation_results': sim_results,
            'qualification_data': qual_data,
            'feature_importance': feature_importance,
            'variability_stats': variability_stats,
            'elapsed_time': elapsed
        }
    
    return results

def display_results(results: Dict, conference: str):
    """Display prediction results."""
    print(f"\n{'='*60}")
    print(f"RESULTS - {conference.upper()} Conference")
    print(f"{'='*60}")
    
    for method, data in results.items():
        if not data:
            continue
            
        print(f"\n📊 {method.upper()} Method:")
        print(f"   Time: {data['elapsed_time']:.1f} seconds")
        
        summary_df = data['summary_df']
        
        # Display top 10 teams
        print(f"\n   Top 10 Teams:")
        print(f"   {'Rank':<5} {'Team':<25} {'Current':<8} {'Playoff %':<10} {'Avg Pts':<8}")
        print(f"   {'-'*5} {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
        
        for idx, row in summary_df.head(10).iterrows():
            print(f"   {idx+1:<5} {row['Team'][:24]:<25} "
                  f"{row['Current Points']:<8} "
                  f"{row['Playoff Qualification %']:<10.1f} "
                  f"{row['Average Points']:<8.1f}")
        
        # Playoff summary
        playoff_teams = summary_df[summary_df['Playoff Qualification %'] >= 99.0]
        bubble_teams = summary_df[
            (summary_df['Playoff Qualification %'] > 20) & 
            (summary_df['Playoff Qualification %'] < 80)
        ]
        eliminated_teams = summary_df[summary_df['Playoff Qualification %'] < 1.0]
        
        print(f"\n   Playoff Picture:")
        print(f"     Clinched (>99%): {len(playoff_teams)} teams")
        print(f"     Bubble (20-80%): {len(bubble_teams)} teams")
        print(f"     Eliminated (<1%): {len(eliminated_teams)} teams")


def compare_methods(results: Dict, conference: str):
    """Compare results between different prediction methods."""
    if len(results) < 2:
        return
    
    print(f"\n{'='*60}")
    print(f"METHOD COMPARISON - {conference.upper()} Conference")
    print(f"{'='*60}")
    
    # Get both methods' data
    mc_data = results.get('monte_carlo', {})
    ml_data = results.get('ml', {})
    
    if not mc_data or not ml_data:
        print("Need both methods for comparison")
        return
    
    mc_df = mc_data['summary_df']
    ml_df = ml_data['summary_df']
    
    # Merge on team ID
    comparison = pd.merge(
        mc_df[['_team_id', 'Team', 'Current Points', 'Playoff Qualification %', 'Average Points']],
        ml_df[['_team_id', 'Playoff Qualification %', 'Average Points']],
        on='_team_id',
        suffixes=('_MC', '_ML')
    )
    
    # Calculate differences
    comparison['Playoff_Diff'] = comparison['Playoff Qualification %_ML'] - comparison['Playoff Qualification %_MC']
    comparison['Points_Diff'] = comparison['Average Points_ML'] - comparison['Average Points_MC']
    
    # Sort by absolute playoff difference
    comparison['Abs_Diff'] = comparison['Playoff_Diff'].abs()
    comparison = comparison.sort_values('Abs_Diff', ascending=False)
    
    print(f"\n{'Team':<25} {'Current':<8} {'MC %':<8} {'ML %':<8} {'Diff':<8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    # Show top differences
    for _, row in comparison.head(10).iterrows():
        arrow = "↑" if row['Playoff_Diff'] > 0 else "↓" if row['Playoff_Diff'] < 0 else "="
        print(f"{row['Team'][:24]:<25} "
              f"{row['Current Points']:<8.0f} "
              f"{row['Playoff Qualification %_MC']:<8.1f} "
              f"{row['Playoff Qualification %_ML']:<8.1f} "
              f"{arrow} {abs(row['Playoff_Diff']):<6.1f}")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Average playoff % difference: {comparison['Playoff_Diff'].mean():.1f}%")
    print(f"   Max increase (ML > MC): {comparison['Playoff_Diff'].max():.1f}%")
    print(f"   Max decrease (ML < MC): {comparison['Playoff_Diff'].min():.1f}%")
    print(f"   Teams with >10% difference: {len(comparison[comparison['Abs_Diff'] > 10])}")


async def save_results(results: Dict, conference: str, season_year: int, choices: Dict):
    """Save results to file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_data = {
        'metadata': {
            'conference': conference,
            'season_year': season_year,
            'timestamp': timestamp,
            'choices': choices
        },
        'results': {}
    }
    
    for method, data in results.items():
        if data:
            output_data['results'][method] = {
                'elapsed_time': data['elapsed_time'],
                'summary': data['summary_df'].to_dict('records'),
                'qualification_data': data['qualification_data']
            }
    
    output_file = f"output/{conference}_predictions_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

def generate_charts_if_requested(results: Dict, conference: str, 
                                n_simulations: int, prediction_method: str):
    """
    Generate charts based on prediction method and available data.
    """
    try:
        chart_generator = MLSNPChartGenerator()
        
        if prediction_method == "both":
            # Create comparison data structure
            comparison_data = {
                'monte_carlo': results.get('monte_carlo'),
                'machine_learning': results.get('ml')
            }
            
            # Get feature importance and variability from ML results
            ml_data = results.get('ml', {})
            feature_importance = ml_data.get('feature_importance', {})
            variability_stats = ml_data.get('variability_stats', {})
            
            chart_files = chart_generator.generate_all_charts(
                summary_df=None,
                simulation_results={},
                qualification_data={},
                conference=conference,
                n_simulations=n_simulations,
                prediction_method="both",
                feature_importance=feature_importance,
                comparison_data=comparison_data,
                variability_stats=variability_stats
            )
            
        elif prediction_method == "ml":
            # ML method only
            data = results['ml']
            feature_importance = data.get('feature_importance', {})
            variability_stats = data.get('variability_stats', {})
            
            chart_files = chart_generator.generate_all_charts(
                summary_df=data['summary_df'],
                simulation_results=data['simulation_results'],
                qualification_data=data['qualification_data'],
                conference=conference,
                n_simulations=n_simulations,
                prediction_method="ml",
                feature_importance=feature_importance,
                variability_stats=variability_stats
            )
            
        else:
            # Monte Carlo method only
            data = results['monte_carlo']
            
            chart_files = chart_generator.generate_all_charts(
                summary_df=data['summary_df'],
                simulation_results=data['simulation_results'],
                qualification_data=data['qualification_data'],
                conference=conference,
                n_simulations=n_simulations,
                prediction_method="monte_carlo"
            )
        
        print(f"🔍 Chart generation completed")
        print(f"    Method: {prediction_method}")
        print(f"    Files created: {list(chart_files.keys())}")
        
        chart_generator.show_charts_summary(chart_files, conference)
        
        # Ask if they want to open the dashboard
        open_dashboard = input("\nOpen dashboard in browser? (y/n): ").lower().strip()
        if open_dashboard == 'y':
            import webbrowser
            dashboards_opened = 0
    
            # Open all dashboard files
            for key, path in chart_files.items():
                if 'dashboard' in key and path:
                    webbrowser.open(f'file://{os.path.abspath(path)}')
                    dashboards_opened += 1
            
            if dashboards_opened > 0:
                print(f"Opened {dashboards_opened} dashboard(s) in browser!")
            else:
                print("No dashboard files found to open.")
        
    except Exception as e:
        logger.error(f"Error generating charts: {e}", exc_info=True)
        print(f"Chart generation failed: {e}")


async def main():
    logger.info("Starting MLS Next Pro Predictor...")
    try:
        # Get user choices
        choices = await get_user_choices()

        # Connect to database
        logger.info("Connecting to database...")
        await database.connect()
        db_manager = DatabaseManager(database)
        await db_manager.initialize() # Ensure conferences and other initial data are set up
        print("✅ Connected!")

        model_manager = MLModelManager(db_manager)

        await update_game_data(
            db_manager, 
            season_year=choices['season_year'], 
            conference=choices['conference']
        )
        
        await db_manager.check_for_rescheduled_games(choices['season_year'])

        # Determine conferences to process
        if choices['conference'] == 'both':
            conferences = ['eastern', 'western']
        else:
            conferences = [choices['conference']]
        
        # Run predictions for each conference
        all_results = {}
        
        for conference in conferences:
            results = await run_predictions_for_conference(
                conference=conference,
                season_year=choices['season_year'],
                n_simulations=choices['n_simulations'],
                prediction_method=choices['prediction_method'],
                db_manager=db_manager,
                model_manager=model_manager,
                calculate_variability=choices.get('calculate_variability', False),
                variability_runs=choices.get('variability_runs', 15)
            )
            
            if results:
                all_results[conference] = results
                
                # Display results
                display_results(results, conference)
                
                # Compare methods if both were run
                if choices['prediction_method'] == 'both':
                    compare_methods(results, conference)
                
                # Save results
                await save_results(results, conference, choices['season_year'], choices)

            # Generate charts if requested
            if choices['generate_charts']:
                generate_charts_if_requested(
                    results=results,
                    conference=conference,
                    n_simulations=choices['n_simulations'],
                    prediction_method=choices['prediction_method'],
                )

        print("\n All predictions completed successfully!")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"❌ An error occurred: {e}")
    
    finally:
        logger.info("Disconnecting from database...")
        await database.disconnect()
    
    logger.info("\nMLS Next Pro Predictor execution finished.")

if __name__ == "__main__":
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    # Run the main function
    asyncio.run(main())
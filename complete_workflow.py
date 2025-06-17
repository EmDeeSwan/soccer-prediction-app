#!/usr/bin/env python3
"""
Complete workflow showing the entire process:
1. Connect to Railway database
2. Update incomplete games from ASA API
3. Train AutoML model (if needed)
4. Run predictions with both traditional and ML methods
5. Compare results
"""

import asyncio
import json
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Apply timezone fix
#import timezone_patch

from src.common.database import database, connect, disconnect
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
import pandas as pd
import numpy as np

async def run_complete_workflow():
    """
    Complete workflow from data to predictions with comparison
    """
    print("🏃 Complete MLS Next Pro Prediction Workflow")
    print("=" * 70)
    
    conference = 'eastern'  # Focus on Eastern conference
    n_simulations = 1000    # Number of simulations to run
    
    try:
        # Step 1: Connect to Railway
        print("\n1️⃣ Connecting to Railway database...")
        await connect()
        db_manager = DatabaseManager(database)
        print("✅ Connected!")
        
        # Step 2: Update game data
        print("\n2️⃣ Checking for game updates...")
        
        # Get current game statistics
        game_stats = await database.fetch_one("""
            SELECT 
                COUNT(*) as total_games,
                SUM(CASE WHEN is_completed THEN 1 ELSE 0 END) as completed_games,
                SUM(CASE WHEN NOT is_completed AND date < NOW() THEN 1 ELSE 0 END) as games_to_update
            FROM games 
            WHERE season_year = 2025
        """)
        
        print(f"   Total games: {game_stats['total_games']}")
        print(f"   Completed: {game_stats['completed_games']}")
        print(f"   Need updates: {game_stats['games_to_update']}")
        
        if game_stats['games_to_update'] > 0:
            print(f"   Updating {game_stats['games_to_update']} games from ASA API...")
            await db_manager.update_incomplete_games(2025)
            print("   ✅ Games updated!")
        
        # Step 3: Load conference data
        print(f"\n3️⃣ Loading {conference} conference data...")
        sim_data = await db_manager.get_data_for_simulation(conference, 2025)
        
        # Calculate league averages from ALL teams
        all_team_xg = []
        for team_id in sim_data['conference_teams']:
            xg_data = await db_manager.get_or_fetch_team_xg(team_id, 2025)
            if xg_data and xg_data.get('games_played', 0) > 0:
                all_team_xg.append({
                    'xgf': xg_data.get('x_goals_for', 0),
                    'xga': xg_data.get('x_goals_against', 0),
                    'games': xg_data.get('games_played', 0)
                })
        
        # Calculate weighted averages
        total_xgf = sum(t['xgf'] for t in all_team_xg)
        total_xga = sum(t['xga'] for t in all_team_xg)
        total_games = sum(t['games'] for t in all_team_xg)
        
        league_averages = {
            'league_avg_xgf': total_xgf / total_games if total_games > 0 else 1.2,
            'league_avg_xga': total_xga / total_games if total_games > 0 else 1.2
        }
        
        print(f"   Teams: {len(sim_data['conference_teams'])}")
        print(f"   League avg xGF: {league_averages['league_avg_xgf']:.2f}")
        print(f"   League avg xGA: {league_averages['league_avg_xga']:.2f}")
        
        # Show current standings (calculate only, don't store)
        # Get all completed games
        games = await db_manager.get_games_for_season(2025, conference, include_incomplete=False)
        
        # Calculate standings manually
        from collections import defaultdict
        standings = defaultdict(lambda: {
            "name": "",
            "points": 0,
            "games_played": 0,
            "wins": 0,
            "goal_difference": 0,
            "goals_for": 0,
            "goals_against": 0
        })
        
        for game in games:
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            home_score = game['home_score'] or 0
            away_score = game['away_score'] or 0
            
            # Set team names
            standings[home_id]["name"] = sim_data['conference_teams'].get(home_id, home_id)
            standings[away_id]["name"] = sim_data['conference_teams'].get(away_id, away_id)
            
            # Update stats
            standings[home_id]["games_played"] += 1
            standings[away_id]["games_played"] += 1
            standings[home_id]["goals_for"] += home_score
            standings[home_id]["goals_against"] += away_score
            standings[away_id]["goals_for"] += away_score
            standings[away_id]["goals_against"] += home_score
            
            # Points calculation
            if game.get('went_to_shootout'):
                # Shootout: winner gets 2, loser gets 1
                home_pens = game.get('home_penalties', 0)
                away_pens = game.get('away_penalties', 0)
                if home_pens > away_pens:
                    standings[home_id]["points"] += 2
                    standings[away_id]["points"] += 1
                else:
                    standings[away_id]["points"] += 2
                    standings[home_id]["points"] += 1
            else:
                # Regular game
                if home_score > away_score:
                    standings[home_id]["points"] += 3
                    standings[home_id]["wins"] += 1
                elif away_score > home_score:
                    standings[away_id]["points"] += 3
                    standings[away_id]["wins"] += 1
                else:
                    # Draw (shouldn't happen in MLS Next Pro)
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1
        
        # Calculate goal difference and sort
        current_standings = []
        for team_id, stats in standings.items():
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
            stats["team_id"] = team_id
            current_standings.append(stats)
        
        current_standings.sort(key=lambda x: (
            -x["points"],
            -x["wins"],
            -x["goal_difference"],
            -x["goals_for"]
        ))
        
        print("\n   Current Top 5:")
        for i, team in enumerate(current_standings[:5], 1):
            print(f"     {i}. {team['name']} - {team['points']} pts")
        
        # Step 4: Run TRADITIONAL predictions
        print(f"\n4️⃣ Running traditional Poisson predictions ({n_simulations} simulations)...")
        
        traditional_predictor = MLSNPRegSeasonPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages,
            use_automl=False  # Force traditional method
        )
        
        trad_summary, trad_ranks, _, trad_qual = traditional_predictor.run_simulations(n_simulations)
        print("✅ Traditional predictions complete!")
        
        # Step 5: Run ML predictions
        print(f"\n5️⃣ Running AutoML predictions ({n_simulations} simulations)...")
        print("   (First run will train the model - this takes ~3 minutes)")
        
        ml_predictor = MLSNPRegSeasonPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages,
            use_automl=True  # Use ML method
        )
        
        if ml_predictor.use_automl:
            ml_summary, ml_ranks, _, ml_qual = ml_predictor.run_simulations(n_simulations)
            print("✅ ML predictions complete!")
        else:
            print("⚠️  AutoML not available, skipping ML predictions")
            ml_summary = None
        
        # Step 6: Compare results
        print("\n6️⃣ COMPARISON: Traditional vs ML Predictions")
        print("=" * 90)
        print(f"{'Team':<30} {'Current':>8} | {'Trad Playoff%':>13} {'ML Playoff%':>13} | {'Difference':>10}")
        print("-" * 90)
        
        # Merge results for comparison
        comparison_data = []
        for idx, trad_row in trad_summary.iterrows():
            team_name = trad_row['Team']
            team_id = trad_row['_team_id']
            
            # Find corresponding ML prediction
            if ml_summary is not None:
                ml_row = ml_summary[ml_summary['_team_id'] == team_id]
                if not ml_row.empty:
                    ml_playoff = ml_row.iloc[0]['Playoff Qualification %']
                else:
                    ml_playoff = 0
            else:
                ml_playoff = trad_row['Playoff Qualification %']  # Use traditional if ML not available
            
            trad_playoff = trad_row['Playoff Qualification %']
            diff = ml_playoff - trad_playoff
            
            comparison_data.append({
                'team': team_name,
                'current_points': trad_row['Current Points'],
                'trad_playoff': trad_playoff,
                'ml_playoff': ml_playoff,
                'difference': diff
            })
        
        # Sort by ML playoff probability
        comparison_data.sort(key=lambda x: x['ml_playoff'], reverse=True)
        
        for i, data in enumerate(comparison_data):
            arrow = "↑" if data['difference'] > 0 else "↓" if data['difference'] < 0 else "="
            print(f"{data['team']:<30} {data['current_points']:>8} | "
                  f"{data['trad_playoff']:>12.1f}% {data['ml_playoff']:>12.1f}% | "
                  f"{arrow} {abs(data['difference']):>8.1f}%")
            
            if i == 7:  # After 8th place
                print("-" * 90 + " ← Playoff Line")
        
        # Step 7: Chattanooga FC Deep Dive
        print("\n7️⃣ CHATTANOOGA FC ANALYSIS")
        print("=" * 70)
        
        # Find CFC in both predictions
        cfc_trad = trad_summary[trad_summary['Team'].str.contains('Chattanooga', case=False)]
        cfc_ml = ml_summary[ml_summary['Team'].str.contains('Chattanooga', case=False)] if ml_summary is not None else None
        
        if not cfc_trad.empty:
            cfc_t = cfc_trad.iloc[0]
            print(f"\n📊 Traditional Model:")
            print(f"   Current Points: {cfc_t['Current Points']}")
            print(f"   Projected Final Points: {cfc_t['Average Points']:.1f}")
            print(f"   Playoff Probability: {cfc_t['Playoff Qualification %']:.1f}%")
            print(f"   Average Final Position: {cfc_t['Average Final Rank']:.1f}")
            
            if cfc_ml is not None and not cfc_ml.empty:
                cfc_m = cfc_ml.iloc[0]
                print(f"\n🤖 ML Model:")
                print(f"   Current Points: {cfc_m['Current Points']}")
                print(f"   Projected Final Points: {cfc_m['Average Points']:.1f}")
                print(f"   Playoff Probability: {cfc_m['Playoff Qualification %']:.1f}%")
                print(f"   Average Final Position: {cfc_m['Average Final Rank']:.1f}")
                
                print(f"\n📈 ML vs Traditional:")
                print(f"   Playoff % difference: {cfc_m['Playoff Qualification %'] - cfc_t['Playoff Qualification %']:+.1f}%")
                print(f"   Points difference: {cfc_m['Average Points'] - cfc_t['Average Points']:+.1f}")
            
            # Show upcoming games
            cfc_id = cfc_t['_team_id']
            remaining_games = [g for g in ml_predictor.remaining_games 
                             if (g['home_team_id'] == cfc_id or g['away_team_id'] == cfc_id)]
            
            print(f"\n📅 Next 5 games:")
            for i, game in enumerate(remaining_games[:5]):
                game_date = game.get('date', 'TBD')
                if hasattr(game_date, 'strftime'):
                    date_str = game_date.strftime('%m/%d')
                else:
                    date_str = str(game_date).split()[0] if game_date else 'TBD'
                
                if game['home_team_id'] == cfc_id:
                    opponent = sim_data['conference_teams'].get(game['away_team_id'], 'Unknown')
                    print(f"   {date_str}: vs {opponent} (H)")
                else:
                    opponent = sim_data['conference_teams'].get(game['home_team_id'], 'Unknown')
                    print(f"   {date_str}: @ {opponent} (A)")
        
        # Step 8: Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"output/comparison_results_{conference}_{timestamp}.json"
        
        results = {
            'run_info': {
                'conference': conference,
                'n_simulations': n_simulations,
                'run_date': datetime.now().isoformat(),
                'completed_games': game_stats['completed_games'],
                'league_averages': league_averages
            },
            'comparison': comparison_data,
            'traditional_results': trad_summary.to_dict('records'),
            'ml_results': ml_summary.to_dict('records') if ml_summary is not None else None,
            'model_info': {
                'ml_available': ml_predictor.use_automl,
                'ml_model_path': 'models/xg_predictor_eastern_202506.pkl' if ml_predictor.use_automl else None
            }
        }
        
        Path(output_file).parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Step 9: Model insights (if ML available)
        if ml_predictor.use_automl and hasattr(ml_predictor, 'ml_model'):
            print("\n9️⃣ ML Model Insights:")
            print(f"   Model type: AutoGluon ensemble")
            print(f"   Training games: {len([g for g in sim_data['games_data'] if g.get('is_completed')])}")
            print(f"   Features used: 22 (including xG, form, H2H, context)")
            
            # If you want to show a sample prediction
            print("\n   Sample prediction (next CFC game):")
            if remaining_games and ml_predictor.use_automl:
                import pandas as pd  # Import pandas here for all the nested functions
                
                next_game = remaining_games[0]
                is_home = next_game['home_team_id'] == cfc_id
                opponent_id = next_game['away_team_id'] if is_home else next_game['home_team_id']
                opponent_name = sim_data['conference_teams'].get(opponent_id, 'Unknown')
                
                # Extract features for the prediction
                game_date_str = next_game['date'].strftime('%Y-%m-%d') if hasattr(next_game['date'], 'strftime') else str(next_game['date']).split()[0]
                games_before = [g for g in sim_data['games_data'] if g.get('is_completed')]
                
                # Patch the date comparison methods temporarily
                original_calculate_h2h = ml_predictor._calculate_h2h_optimized
                original_calculate_rest = ml_predictor._calculate_rest_days_optimized
                
                def patched_calculate_h2h(team_id, opponent_id, before_date, games):
                    import pandas as pd
                    # Convert before_date to datetime if string
                    if isinstance(before_date, str):
                        before_date_dt = pd.to_datetime(before_date)
                    else:
                        before_date_dt = before_date
                    
                    h2h_games = []
                    for game in games:
                        if not game.get('is_completed'):
                            continue
                        
                        game_date = game.get('date')
                        if game_date:
                            game_date_dt = pd.to_datetime(game_date) if isinstance(game_date, str) else game_date
                            if game_date_dt >= before_date_dt:
                                continue
                        
                        # Check if it's a H2H game
                        if ((game['home_team_id'] == team_id and game['away_team_id'] == opponent_id) or
                            (game['home_team_id'] == opponent_id and game['away_team_id'] == team_id)):
                            h2h_games.append(game)
                    
                    # Calculate H2H features
                    if not h2h_games:
                        return {
                            'h2h_games_played': 0,
                            'h2h_win_rate': 0.0,
                            'h2h_goals_for_avg': 0.0,
                            'h2h_goals_against_avg': 0.0
                        }
                    
                    wins = 0
                    total_gf = 0
                    total_ga = 0
                    
                    for game in h2h_games:
                        if game['home_team_id'] == team_id:
                            gf = game.get('home_score', 0)
                            ga = game.get('away_score', 0)
                        else:
                            gf = game.get('away_score', 0)
                            ga = game.get('home_score', 0)
                        
                        total_gf += gf
                        total_ga += ga
                        if gf > ga:
                            wins += 1
                    
                    return {
                        'h2h_games_played': len(h2h_games),
                        'h2h_win_rate': wins / len(h2h_games),
                        'h2h_goals_for_avg': total_gf / len(h2h_games),
                        'h2h_goals_against_avg': total_ga / len(h2h_games)
                    }
                
                def patched_calculate_rest_days(team_id, before_date, games):
                    import pandas as pd
                    # Convert before_date to datetime if string
                    if isinstance(before_date, str):
                        before_date_dt = pd.to_datetime(before_date)
                    else:
                        before_date_dt = before_date
                    
                    # Find last game
                    last_game_date = None
                    for game in reversed(games):
                        if not game.get('is_completed'):
                            continue
                        
                        if game['home_team_id'] == team_id or game['away_team_id'] == team_id:
                            game_date = game.get('date')
                            if game_date:
                                game_date_dt = pd.to_datetime(game_date) if isinstance(game_date, str) else game_date
                                if game_date_dt < before_date_dt:
                                    last_game_date = game_date_dt
                                    break
                    
                    if last_game_date:
                        days_rest = (before_date_dt - last_game_date).days
                        return min(days_rest, 14)  # Cap at 14 days
                    
                    return 7  # Default if no previous games
                
                # Apply patches
                ml_predictor._calculate_h2h_features = patched_calculate_h2h
                ml_predictor._calculate_rest_days = patched_calculate_rest_days
                
                # Get features for CFC
                cfc_features = ml_predictor._extract_features(
                    team_id=cfc_id,
                    opponent_id=opponent_id,
                    is_home=is_home,
                    game_date=game_date_str,
                    games_before=games_before
                )
                
                # Get features for opponent
                opp_features = ml_predictor._extract_features(
                    team_id=opponent_id,
                    opponent_id=cfc_id,
                    is_home=not is_home,
                    game_date=game_date_str,
                    games_before=games_before
                )
                
                # Make predictions
                import pandas as pd
                cfc_pred_df = pd.DataFrame([cfc_features])
                opp_pred_df = pd.DataFrame([opp_features])
                
                # Remove league averages if present (not used by model)
                for col in ['league_avg_xgf', 'league_avg_xga']:
                    if col in cfc_pred_df.columns:
                        cfc_pred_df = cfc_pred_df.drop([col], axis=1)
                    if col in opp_pred_df.columns:
                        opp_pred_df = opp_pred_df.drop([col], axis=1)
                
                # Get predictions
                cfc_predicted_goals = ml_predictor.ml_model.predict(cfc_pred_df)[0]
                opp_predicted_goals = ml_predictor.ml_model.predict(opp_pred_df)[0]
                
                # Also get traditional xG prediction for comparison
                cfc_xg = cfc_features.get('team_xgf_per_game', 1.2) * cfc_features.get('opp_xga_per_game', 1.2) / league_averages['league_avg_xga']
                opp_xg = opp_features.get('team_xgf_per_game', 1.2) * opp_features.get('opp_xga_per_game', 1.2) / league_averages['league_avg_xga']
                
                # Adjust for home advantage if CFC is home
                if is_home:
                    cfc_xg *= 1.1  # 10% home boost
                else:
                    opp_xg *= 1.1
                
                print(f"\n   📊 Next Game Prediction:")
                print(f"   {game_date_str}: {'Chattanooga FC' if is_home else opponent_name} vs {opponent_name if is_home else 'Chattanooga FC'}")
                print(f"   ")
                print(f"   Traditional xG Model:")
                print(f"     Chattanooga FC: {cfc_xg:.2f} xG")
                print(f"     {opponent_name}: {opp_xg:.2f} xG")
                print(f"   ")
                print(f"   ML Model Prediction:")
                print(f"     Chattanooga FC: {cfc_predicted_goals:.2f} goals")
                print(f"     {opponent_name}: {opp_predicted_goals:.2f} goals")
                print(f"   ")
                print(f"   📈 Analysis:")
                if cfc_predicted_goals > opp_predicted_goals:
                    win_margin = cfc_predicted_goals - opp_predicted_goals
                    print(f"     ML predicts CFC win by {win_margin:.1f} goals")
                elif opp_predicted_goals > cfc_predicted_goals:
                    loss_margin = opp_predicted_goals - cfc_predicted_goals
                    print(f"     ML predicts CFC loss by {loss_margin:.1f} goals")
                else:
                    print(f"     ML predicts a draw")
                
                # Win probability estimate (simplified)
                goal_diff = cfc_predicted_goals - opp_predicted_goals
                win_prob = 1 / (1 + np.exp(-goal_diff * 0.5)) * 100  # Sigmoid function
                print(f"     Estimated win probability: {win_prob:.1f}%")
        
        print("\n✅ Workflow complete!")
        print("\n💡 Key Insights:")
        print("   - ML model considers actual performance, not just xG")
        print("   - Traditional model may over/underestimate based on xG alone")
        print("   - Biggest differences show where performance != expected goals")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await disconnect()


if __name__ == "__main__":
    print("🚀 MLS Next Pro Complete Prediction Workflow")
    print("=" * 50)
    print("This workflow will:")
    print("  1. Connect to Railway database")
    print("  2. Update any incomplete games")
    print("  3. Run traditional Poisson predictions")
    print("  4. Train/use AutoML model for predictions")
    print("  5. Compare both methods")
    print("  6. Analyze Chattanooga FC specifically")
    print("\nEstimated time: 3-5 minutes")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    asyncio.run(run_complete_workflow())


import asyncio
import json
import os
from databases import Database
from dotenv import load_dotenv
from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.advanced_predictor import SingleMatchPredictor

async def main():
    """
    This script allows you to predict the outcome of a single match between two teams.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL_PUBLIC")
    if not db_url:
        print("DATABASE_URL_PUBLIC environment variable not set.")
        return

    try:
        async with Database(db_url) as database:
            db_manager = DatabaseManager(database)

            # Fetch teams from the Eastern conference
            eastern_teams = await db_manager.get_conference_teams(1, 2025)
            
            # Let the user choose the teams
            print("Eastern Conference Teams:")
            for i, (team_id, team_name) in enumerate(eastern_teams.items()):
                print(f"{i+1}. {team_name} ({team_id})")

            while True:
                try:
                    choice1 = int(input("Choose the first team (by number): "))
                    choice2 = int(input("Choose the second team (by number): "))
                    if 1 <= choice1 <= len(eastern_teams) and 1 <= choice2 <= len(eastern_teams) and choice1 != choice2:
                        break
                    else:
                        print("Invalid choice. Please choose two different teams from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            sorted_teams = sorted(eastern_teams.items(), key=lambda x: x[1])
            team1_id = sorted_teams[choice1 - 1][0]
            team2_id = sorted_teams[choice2 - 1][0]
            
            # Instantiate the predictor
            predictor = SingleMatchPredictor(db_manager)

            # Predict the match
            prediction = await predictor.predict_match(team1_id, team2_id, model_type='both')
            team_data = await predictor.get_team_performance_data(team1_id, team2_id)

            # Print the results in a human-readable format
            print(f"\nMatch Prediction: {eastern_teams[team1_id]} vs {eastern_teams[team2_id]}")
            print("=" * 60)
            
                        # Team performance data
            home_team_xg_per_game = team_data[team1_id]["x_goals_for"] / team_data[team1_id]["games_played"]
            home_team_xga_per_game = team_data[team1_id]["x_goals_against"] / team_data[team1_id]["games_played"]
            away_team_xg_per_game = team_data[team2_id]["x_goals_for"] / team_data[team2_id]["games_played"]
            away_team_xga_per_game = team_data[team2_id]["x_goals_against"] / team_data[team2_id]["games_played"]

            print("\nTeam Performance Metrics (per game):")
            print(f"\n{eastern_teams[team1_id]} (Home):")
            print(f"Offensive Strength: {home_team_xg_per_game:.2f} xG For")
            print(f"Defensive Strength: {home_team_xga_per_game:.2f} xG Against")
            print(f"\n{eastern_teams[team2_id]} (Away):")
            print(f"Offensive Strength: {away_team_xg_per_game:.2f} xG For")
            print(f"Defensive Strength: {away_team_xga_per_game:.2f} xG Against")
            
            # Monte Carlo predictions
            mc_results = prediction['monte_carlo_results']

            print("\nMonte Carlo Model Predictions:")
            print(f"{eastern_teams[team1_id]} win: {mc_results['home_win_regulation']:.1%}")
            print(f"{eastern_teams[team2_id]} win: {mc_results['away_win_regulation']:.1%}")
            print(f"Draw: {mc_results['draw_regulation']:.1%}")
            
            # ML model predictions
            ml_results = prediction['ml_results']
            print("\nML Model Predictions:")
            print(f"{eastern_teams[team1_id]} win: {ml_results['home_win_regulation']:.1%}")
            print(f"{eastern_teams[team2_id]} win: {ml_results['away_win_regulation']:.1%}")
            print(f"Draw: {ml_results['draw_regulation']:.1%}")
            
            # Shootout probabilities for points
            print("\nIf the match ends in a draw:")
            print("Shootout Points Distribution:")
            so_home_prob = mc_results['home_win_shootout_given_draw']
            so_away_prob = mc_results['away_win_shootout_given_draw']
            print(f"{eastern_teams[team1_id]} wins shootout: {so_home_prob:.1%}")
            print(f"{eastern_teams[team2_id]} wins shootout: {so_away_prob:.1%}")
            
            # Expected standings points
            draw_prob = mc_results['draw_regulation']
            home_win_prob = mc_results['home_win_regulation']
            away_win_prob = mc_results['away_win_regulation']
            
            home_expected_points = (
                home_win_prob * 3 +  # Win = 3 standings points
                draw_prob * (so_home_prob * 2 + (1 - so_home_prob) * 1)  # Draw + shootout
            )
            away_expected_points = (
                away_win_prob * 3 +  # Win = 3 standings points
                draw_prob * (so_away_prob * 2 + (1 - so_away_prob) * 1)  # Draw + shootout
            )
            
            print(f"\nExpected Standings Points (from Monte Carlo simulation):")
            print(f"{eastern_teams[team1_id]}: {home_expected_points:.2f}")
            print(f"{eastern_teams[team2_id]}: {away_expected_points:.2f}")
            
            # Most likely scores
            print("\nScore Predictions:")
            
            # ML model prediction
            ml_home_score, ml_away_score = prediction['potential_scores']['ml'].split('-')
            print(f"\nML Model Expected Score:")
            print(f"{eastern_teams[team1_id]} {ml_home_score} - {ml_away_score} {eastern_teams[team2_id]}")
            
            print("\nMonte Carlo Most Likely Scores:")
            
            # Overall top 5
            print("\nTop 5 Most Likely Scores Overall:")
            scores = prediction['potential_scores']['monte_carlo']
            for score, count in scores[:5]:
                percentage = count / prediction['n_simulations'] * 100
                home_score, away_score = score.split('-')
                print(f"{eastern_teams[team1_id]} {home_score} - {away_score} {eastern_teams[team2_id]}: {percentage:.1f}%")
            
            # Home team wins
            print(f"\nMost Likely {eastern_teams[team1_id]} Wins:")
            # Home team wins - explicitly check all scores
            home_wins = []
            for score, count in scores:
                home_goals, away_goals = map(int, score.split('-'))
                if home_goals > away_goals:
                    home_wins.append((score, count))
            if home_wins:
                for score, count in sorted(home_wins, key=lambda x: x[1], reverse=True)[:5]:
                    percentage = count / prediction['n_simulations'] * 100
                    home_score, away_score = score.split('-')
                    print(f"{eastern_teams[team1_id]} {home_score} - {away_score} {eastern_teams[team2_id]}: {percentage:.1f}%")
            else:
                print("No wins simulated for", eastern_teams[team1_id])
            
            # Away team wins
            print(f"\nMost Likely {eastern_teams[team2_id]} Wins:")
            # Away team wins - explicitly check all scores
            away_wins = []
            for score, count in scores:
                home_goals, away_goals = map(int, score.split('-'))
                if away_goals > home_goals:
                    away_wins.append((score, count))
            if away_wins:
                for score, count in sorted(away_wins, key=lambda x: x[1], reverse=True)[:5]:
                    percentage = count / prediction['n_simulations'] * 100
                    home_score, away_score = score.split('-')
                    print(f"{eastern_teams[team1_id]} {home_score} - {away_score} {eastern_teams[team2_id]}: {percentage:.1f}%")
            else:
                print("No wins simulated for", eastern_teams[team2_id])
                
            # Draws
            print(f"\nMost Likely Draws:")
            # Draws - explicitly check all scores
            draws = []
            for score, count in scores:
                home_goals, away_goals = map(int, score.split('-'))
                if home_goals == away_goals:
                    draws.append((score, count))
            if draws:
                for score, count in sorted(draws, key=lambda x: x[1], reverse=True)[:5]:
                    percentage = count / prediction['n_simulations'] * 100
                    home_score, away_score = score.split('-')
                    print(f"{eastern_teams[team1_id]} {home_score} - {away_score} {eastern_teams[team2_id]}: {percentage:.1f}%")
            else:
                print("No draws simulated")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print("Full error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())

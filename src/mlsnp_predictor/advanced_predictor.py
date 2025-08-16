import logging
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
import copy
from scipy import stats

from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.reg_season_predictor import MLSNPRegSeasonPredictor
from src.mlsnp_predictor.machine_learning import MLPredictor

logger = logging.getLogger(__name__)

class SingleMatchPredictor:
    """
    Predicts the outcome of a single match using different models.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.team_to_conference = None

    async def _get_team_conference(self, team_id: str) -> str:
        if self.team_to_conference is None:
            self.team_to_conference = {}
            for conf_id, conf_name in [(1, "eastern"), (2, "western")]:
                teams = await self.db_manager.get_conference_teams(conf_id, 2025)
                for team in teams:
                    self.team_to_conference[team['id']] = conf_name
        return self.team_to_conference.get(team_id)

    async def predict_match(self, team1_id: str, team2_id: str, model_type: str = 'monte_carlo', n_simulations: int = 10000) -> Dict[str, Any]:
        """
        Predicts the outcome of a single match.
        """
        conference1 = await self._get_team_conference(team1_id)
        conference2 = await self._get_team_conference(team2_id)

        if conference1 != conference2 or conference1 is None:
            raise ValueError("Both teams must be in the same, valid conference for prediction.")

        conference = conference1

        if model_type == 'monte_carlo':
            return await self._predict_with_monte_carlo(team1_id, team2_id, conference, n_simulations)
        elif model_type == 'ml':
            return await self._predict_with_ml(team1_id, team2_id, conference)
        elif model_type == 'both':
            import asyncio
            mc_results_task = self._predict_with_monte_carlo(team1_id, team2_id, conference, n_simulations)
            ml_results_task = self._predict_with_ml(team1_id, team2_id, conference)

            mc_results, ml_results = await asyncio.gather(mc_results_task, ml_results_task)

            combined_probs = {}
            for key in mc_results['probabilities']:
                mc_prob = mc_results['probabilities'][key]
                ml_prob = ml_results['probabilities'].get(key, 0)
                combined_probs[key] = (mc_prob + ml_prob) / 2

            return {
                "probabilities": combined_probs,
                "potential_scores": {
                    "monte_carlo": mc_results['potential_scores'],
                    "ml": ml_results['potential_score']
                },
                "n_simulations": n_simulations,
                "model_version": ml_results.get('model_version')
            }
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    async def _predict_with_monte_carlo(self, home_team_id: str, away_team_id: str, conference: str, n_simulations: int) -> Dict[str, Any]:
        sim_data = await self.db_manager.get_data_for_simulation(conference, 2025)
        league_averages = await self._calculate_league_averages()

        predictor = MLSNPRegSeasonPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages
        )

        game = {"home_team_id": home_team_id, "away_team_id": away_team_id}

        outcomes = defaultdict(int)
        scores = defaultdict(int)

        for _ in range(n_simulations):
            h_goals, a_goals, went_to_shootout, home_wins_shootout = predictor._simulate_game(game)

            score_key = f"{h_goals}-{a_goals}"
            scores[score_key] += 1

            if went_to_shootout:
                outcomes['draw'] += 1
                if home_wins_shootout:
                    outcomes['home_win_shootout'] += 1
                else:
                    outcomes['away_win_shootout'] += 1
            elif h_goals > a_goals:
                outcomes['home_win'] += 1
            else:
                outcomes['away_win'] += 1

        total_sims = n_simulations
        draw_prob = outcomes['draw'] / total_sims
        home_win_reg_prob = outcomes['home_win'] / total_sims
        away_win_reg_prob = outcomes['away_win'] / total_sims

        home_so_win_given_draw_prob = outcomes['home_win_shootout'] / outcomes['draw'] if outcomes['draw'] > 0 else 0

        home_win_total = (outcomes['home_win'] + outcomes['home_win_shootout']) / total_sims
        away_win_total = (outcomes['away_win'] + outcomes['away_win_shootout']) / total_sims

        results = {
            "probabilities": {
                "home_win_regulation": home_win_reg_prob,
                "away_win_regulation": away_win_reg_prob,
                "draw_regulation": draw_prob,
                "home_win_shootout_given_draw": home_so_win_given_draw_prob,
                "away_win_shootout_given_draw": 1 - home_so_win_given_draw_prob if outcomes['draw'] > 0 else 0,
                "home_win_total": home_win_total,
                "away_win_total": away_win_total,
            },
            "potential_scores": sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5],
            "n_simulations": n_simulations
        }

        return results

    async def _predict_with_ml(self, home_team_id: str, away_team_id: str, conference: str) -> Dict[str, Any]:
        sim_data = await self.db_manager.get_data_for_simulation(conference, 2025)
        league_averages = await self._calculate_league_averages()

        predictor = MLPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages
        )

        if not predictor.ml_model:
            raise RuntimeError("ML model is not trained or loaded. Cannot make ML prediction.")

        home_features = predictor._extract_features(home_team_id, away_team_id, True)
        away_features = predictor._extract_features(away_team_id, home_team_id, False)

        if predictor.AUTOML_LIBRARY == "autogluon":
            import pandas as pd
            home_exp_goals = predictor.ml_model.predict(pd.DataFrame([home_features]))[0]
            away_exp_goals = predictor.ml_model.predict(pd.DataFrame([away_features]))[0]
        else: # sklearn
            import pandas as pd
            home_X = pd.DataFrame([home_features])[predictor._feature_names]
            away_X = pd.DataFrame([away_features])[predictor._feature_names]
            home_exp_goals = predictor.ml_model.predict(home_X)[0]
            away_exp_goals = predictor.ml_model.predict(away_X)[0]

        home_exp_goals = max(0.1, home_exp_goals)
        away_exp_goals = max(0.1, away_exp_goals)

        home_win_prob = predictor._calculate_win_probability(home_exp_goals, away_exp_goals)
        away_win_prob = predictor._calculate_win_probability(away_exp_goals, home_exp_goals)
        draw_prob = 1.0 - home_win_prob - away_win_prob

        home_shootout_win_prob_in_draw = 0.55

        return {
            "probabilities": {
                "home_win_total": home_win_prob,
                "away_win_total": away_win_prob,
                "draw_regulation": draw_prob,
                "home_win_shootout": home_shootout_win_prob_in_draw,
                "away_win_shootout": 1 - home_shootout_win_prob_in_draw,
                "home_win_regulation": home_win_prob - (draw_prob * home_shootout_win_prob_in_draw),
                "away_win_regulation": away_win_prob - (draw_prob * (1-home_shootout_win_prob_in_draw)),
            },
            "potential_score": f"{home_exp_goals:.2f}-{away_exp_goals:.2f}",
            "model_version": predictor.model_version
        }

    async def _calculate_league_averages(self) -> Dict[str, float]:
        all_teams_xg = await self.db_manager.db.fetch_all("""
            SELECT
                team_id,
                x_goals_for,
                x_goals_against,
                games_played
            FROM team_xg_history
            WHERE season_year = 2025
            AND games_played > 0
            ORDER BY team_id, date_captured DESC
        """)

        team_latest = {}
        for row in all_teams_xg:
            if row['team_id'] not in team_latest:
                team_latest[row['team_id']] = row

        total_xgf = sum(t['x_goals_for'] for t in team_latest.values())
        total_xga = sum(t['x_goals_against'] for t in team_latest.values())
        total_games = sum(t['games_played'] for t in team_latest.values())

        if total_games > 0:
            league_avg_xgf = total_xgf / total_games
            league_avg_xga = total_xga / total_games
        else:
            league_avg_xgf = 1.2
            league_avg_xga = 1.2

        return {
            "league_avg_xgf": league_avg_xgf,
            "league_avg_xga": league_avg_xga,
        }

class EndSeasonScenarioGenerator:
    """
    Generates end-of-season scenarios for a given team.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.team_to_conference = None

    async def _get_team_conference(self, team_id: str) -> str:
        if self.team_to_conference is None:
            self.team_to_conference = {}
            for conf_id, conf_name in [(1, "eastern"), (2, "western")]:
                teams = await self.db_manager.get_conference_teams(conf_id, 2025)
                for team in teams:
                    self.team_to_conference[team['id']] = conf_name
        return self.team_to_conference.get(team_id)

    async def generate_scenarios(self, team_id: str) -> Dict[str, Any]:
        conference = await self._get_team_conference(team_id)
        if not conference:
            raise ValueError(f"Could not determine conference for team {team_id}")

        sim_data = await self.db_manager.get_data_for_simulation(conference, 2025)

        all_conference_teams = {t_id for t_id in sim_data["conference_teams"].keys()}

        remaining_games = [
            g for g in sim_data["games_data"]
            if not g.get("is_completed") and
               g.get("home_team_id") in all_conference_teams and
               g.get("away_team_id") in all_conference_teams
        ]

        if len(remaining_games) > 5:
            return {
                "error": "Too many games remaining (> 5). Scenario generation is not feasible.",
                "games_remaining": len(remaining_games)
            }

        if not remaining_games:
            return {"message": "No games remaining in the season."}

        league_averages = await self._calculate_league_averages(all_conference_teams)
        ml_predictor = MLPredictor(
            conference=conference,
            conference_teams=sim_data["conference_teams"],
            games_data=sim_data["games_data"],
            team_performance=sim_data["team_performance"],
            league_averages=league_averages
        )
        if not ml_predictor.ml_model:
            raise RuntimeError("ML model is not trained or loaded. Cannot generate scenarios.")

        game_probabilities = {}
        for game in remaining_games:
            game_id = game['id']
            game_probabilities[game_id] = self._get_game_probabilities(game, ml_predictor)

        initial_standings = self._calculate_current_standings(sim_data["games_data"], all_conference_teams, sim_data["conference_teams"])
        team_names = {t_id: t_name for t_id, t_name in sim_data["conference_teams"].items()}

        results = defaultdict(list)
        await self._find_scenarios(
            remaining_games,
            initial_standings,
            path=[],
            probability=1.0,
            target_team_id=team_id,
            game_probabilities=game_probabilities,
            team_names=team_names,
            results=results
        )

        return self._format_results(results, team_names.get(team_id, team_id))

    def _get_reg_outcome_probs(self, home_exp_goals, away_exp_goals):
        max_goals = 10
        p_hwr = 0
        p_awr = 0
        p_draw = 0
        for h_goals in range(max_goals):
            for a_goals in range(max_goals):
                prob = stats.poisson.pmf(h_goals, home_exp_goals) * stats.poisson.pmf(a_goals, away_exp_goals)
                if h_goals > a_goals:
                    p_hwr += prob
                elif a_goals > h_goals:
                    p_awr += prob
                else:
                    p_draw += prob

        total_p = p_hwr + p_awr + p_draw
        if total_p == 0: return 0, 0, 0 # Avoid division by zero
        return p_hwr / total_p, p_awr / total_p, p_draw / total_p

    def _get_game_probabilities(self, game: Dict, predictor: MLPredictor) -> Dict[str, float]:
        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]

        home_features = predictor._extract_features(home_team_id, away_team_id, True)
        away_features = predictor._extract_features(away_team_id, home_team_id, False)

        if predictor.AUTOML_LIBRARY == "autogluon":
            import pandas as pd
            home_exp_goals = predictor.ml_model.predict(pd.DataFrame([home_features]))[0]
            away_exp_goals = predictor.ml_model.predict(pd.DataFrame([away_features]))[0]
        else:
            import pandas as pd
            home_X = pd.DataFrame([home_features])[predictor._feature_names]
            away_X = pd.DataFrame([away_features])[predictor._feature_names]
            home_exp_goals = predictor.ml_model.predict(home_X)[0]
            away_exp_goals = predictor.ml_model.predict(away_X)[0]

        home_exp_goals = max(0.1, home_exp_goals)
        away_exp_goals = max(0.1, away_exp_goals)

        p_hwr, p_awr, p_draw = self._get_reg_outcome_probs(home_exp_goals, away_exp_goals)

        home_shootout_win_prob_in_draw = 0.55

        return {
            "home_win_reg": p_hwr,
            "away_win_reg": p_awr,
            "home_win_so": p_draw * home_shootout_win_prob_in_draw,
            "away_win_so": p_draw * (1 - home_shootout_win_prob_in_draw),
        }

    async def _find_scenarios(self, games, standings, path, probability, target_team_id, game_probabilities, team_names, results):
        if not games:
            final_rank = self._calculate_final_rank(standings, target_team_id)
            results[final_rank].append({"path": path, "probability": probability})
            return

        game = games[0]
        remaining_games = games[1:]

        probs = game_probabilities[game['id']]
        home_id, away_id = game['home_team_id'], game['away_team_id']

        # Scenario 1: Home wins regulation
        if probs['home_win_reg'] > 0:
            new_standings = self._apply_result(standings, home_id, away_id, "win")
            new_path = path + [f"{team_names.get(home_id)} beats {team_names.get(away_id)}"]
            await self._find_scenarios(remaining_games, new_standings, new_path, probability * probs['home_win_reg'], target_team_id, game_probabilities, team_names, results)

        # Scenario 2: Away wins regulation
        if probs['away_win_reg'] > 0:
            new_standings = self._apply_result(standings, away_id, home_id, "win")
            new_path = path + [f"{team_names.get(away_id)} beats {team_names.get(home_id)}"]
            await self._find_scenarios(remaining_games, new_standings, new_path, probability * probs['away_win_reg'], target_team_id, game_probabilities, team_names, results)

        # Scenario 3: Home wins shootout
        if probs['home_win_so'] > 0:
            new_standings = self._apply_result(standings, home_id, away_id, "so_win")
            new_path = path + [f"{team_names.get(home_id)} beats {team_names.get(away_id)} in SO"]
            await self._find_scenarios(remaining_games, new_standings, new_path, probability * probs['home_win_so'], target_team_id, game_probabilities, team_names, results)

        # Scenario 4: Away wins shootout
        if probs['away_win_so'] > 0:
            new_standings = self._apply_result(standings, away_id, home_id, "so_win")
            new_path = path + [f"{team_names.get(away_id)} beats {team_names.get(home_id)} in SO"]
            await self._find_scenarios(remaining_games, new_standings, new_path, probability * probs['away_win_so'], target_team_id, game_probabilities, team_names, results)

    def _apply_result(self, original_standings, winner_id, loser_id, result_type):
        standings = copy.deepcopy(original_standings)

        winner_stats = standings[winner_id]
        loser_stats = standings[loser_id]

        winner_stats["games_played"] += 1
        loser_stats["games_played"] += 1

        if result_type == "win":
            winner_stats["wins"] += 1
            winner_stats["points"] += 3
            loser_stats["losses"] += 1
        elif result_type == "so_win":
            winner_stats["draws"] += 1
            winner_stats["shootout_wins"] += 1
            winner_stats["points"] += 2
            loser_stats["draws"] += 1
            loser_stats["points"] += 1

        return standings

    def _calculate_final_rank(self, standings, target_team_id):
        sorted_teams = sorted(
            standings.values(),
            key=lambda x: (-x['points'], -x['wins'], -x['goal_difference'], -x['goals_for'], -x['shootout_wins'])
        )
        for rank, stats in enumerate(sorted_teams, 1):
            if stats['team_id'] == target_team_id:
                return rank
        return -1

    def _format_results(self, results, target_team_name):
        formatted = {
            "target_team": target_team_name,
            "scenario_summary": []
        }
        for rank, scenarios in sorted(results.items()):
            total_prob = sum(s['probability'] for s in scenarios)
            formatted["scenario_summary"].append({
                "final_rank": rank,
                "total_probability": total_prob,
                "example_scenarios": scenarios[:3] # Show a few examples
            })
        return formatted

    def _calculate_current_standings(self, games_data, conference_teams, team_names_map):
        standings = defaultdict(lambda: {
            "team_id": None, "name": "", "points": 0, "goal_difference": 0,
            "games_played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "shootout_wins": 0
        })

        for team_id in conference_teams:
            standings[team_id]["team_id"] = team_id
            standings[team_id]["name"] = team_names_map.get(team_id, f"Team {team_id}")

        for game in games_data:
            if not game.get("is_completed", False):
                continue

            home_id, away_id = game["home_team_id"], game["away_team_id"]

            if home_id not in conference_teams or away_id not in conference_teams:
                continue

            home_score = int(game.get("home_score", 0))
            away_score = int(game.get("away_score", 0))

            # This is a simplified version of the logic in MLSNPRegSeasonPredictor
            # It doesn't handle goal difference, which is fine for this purpose
            # as we only need points, wins, and SO wins for tiebreakers.
            # A more robust implementation would share the standings calculation logic.

            standings[home_id]['games_played'] += 1
            standings[away_id]['games_played'] += 1
            standings[home_id]['goals_for'] += home_score
            standings[home_id]['goals_against'] += away_score
            standings[away_id]['goals_for'] += away_score
            standings[away_id]['goals_against'] += home_score

            if game.get("went_to_shootout", False):
                standings[home_id]['draws'] += 1
                standings[away_id]['draws'] += 1
                if game.get("home_penalties", 0) > game.get("away_penalties", 0):
                    standings[home_id]["shootout_wins"] += 1
                    standings[home_id]["points"] += 2
                    standings[away_id]["points"] += 1
                else:
                    standings[away_id]["shootout_wins"] += 1
                    standings[away_id]["points"] += 2
                    standings[home_id]["points"] += 1
            else:
                if home_score > away_score:
                    standings[home_id]['wins'] += 1
                    standings[home_id]['points'] += 3
                    standings[away_id]['losses'] += 1
                elif away_score > home_score:
                    standings[away_id]['wins'] += 1
                    standings[away_id]['points'] += 3
                    standings[home_id]['losses'] += 1
                else: # Should not happen in MLSNP
                    standings[home_id]['draws'] += 1
                    standings[away_id]['draws'] += 1
                    standings[home_id]['points'] += 1
                    standings[away_id]['points'] += 1

        for team_id in standings:
            standings[team_id]['goal_difference'] = standings[team_id]['goals_for'] - standings[team_id]['goals_against']

        return {team_id: dict(stats) for team_id, stats in standings.items()}

    async def _calculate_league_averages(self, all_conference_teams: set) -> Dict[str, float]:
        # Duplicated from SingleMatchPredictor, consider refactoring to a common utility
        all_teams_xg = await self.db_manager.db.fetch_all("""
            SELECT team_id, x_goals_for, x_goals_against, games_played
            FROM team_xg_history WHERE season_year = 2025 AND games_played > 0
            ORDER BY team_id, date_captured DESC
        """)

        team_latest = {}
        for row in all_teams_xg:
            if row['team_id'] not in team_latest:
                team_latest[row['team_id']] = row

        total_xgf = sum(t['x_goals_for'] for t in team_latest.values() if t['team_id'] in all_conference_teams)
        total_xga = sum(t['x_goals_against'] for t in team_latest.values() if t['team_id'] in all_conference_teams)
        total_games = sum(t['games_played'] for t in team_latest.values() if t['team_id'] in all_conference_teams)

        if total_games > 0:
            return {
                "league_avg_xgf": total_xgf / total_games,
                "league_avg_xga": total_xga / total_games
            }
        return {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}

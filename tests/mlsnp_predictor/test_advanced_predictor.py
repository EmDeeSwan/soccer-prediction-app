import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mlsnp_predictor.advanced_predictor import SingleMatchPredictor, EndSeasonScenarioGenerator
from src.common.database_manager import DatabaseManager

@pytest.fixture
def mock_db_manager():
    """Mocks the DatabaseManager."""
    db_manager = MagicMock(spec=DatabaseManager)
    db_manager.get_conference_teams = AsyncMock()
    db_manager.get_data_for_simulation = AsyncMock()
    return db_manager

@pytest.fixture
def sample_sim_data():
    """Provides sample simulation data."""
    return {
        "conference_teams": {
            "team1": "Team One",
            "team2": "Team Two",
            "team3": "Team Three",
        },
        "games_data": [
            # Add some completed games for standings
            {
                "id": 1, "home_team_id": "team1", "away_team_id": "team2", "is_completed": True,
                "home_score": 2, "away_score": 1, "went_to_shootout": False
            },
            # Add some remaining games
            {
                "id": 2, "home_team_id": "team1", "away_team_id": "team3", "is_completed": False
            }
        ],
        "team_performance": {
            "team1": {"x_goals_for": 1.5, "x_goals_against": 1.0, "games_played": 1},
            "team2": {"x_goals_for": 1.0, "x_goals_against": 1.5, "games_played": 1},
            "team3": {"x_goals_for": 1.2, "x_goals_against": 1.2, "games_played": 0},
        }
    }

@pytest.mark.asyncio
class TestSingleMatchPredictor:
    async def test_predict_match_monte_carlo(self, mock_db_manager, sample_sim_data):
        # Arrange
        mock_db_manager.get_conference_teams.return_value = [
            {'id': 'team1', 'name': 'Team One'},
            {'id': 'team2', 'name': 'Team Two'},
            {'id': 'team3', 'name': 'Team Three'},
        ]
        mock_db_manager.get_data_for_simulation.return_value = sample_sim_data

        # Mock the league averages calculation
        with patch('src.mlsnp_predictor.advanced_predictor.SingleMatchPredictor._calculate_league_averages', new_callable=AsyncMock) as mock_avg:
            mock_avg.return_value = {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}

            predictor = SingleMatchPredictor(mock_db_manager)

            # Act
            result = await predictor.predict_match("team1", "team2", model_type='monte_carlo', n_simulations=100)

            # Assert
            assert "probabilities" in result
            assert "potential_scores" in result
            assert result["n_simulations"] == 100
            assert "home_win_total" in result["probabilities"]
            # The sum of total win probs and draw prob should be ~1
            total_prob = result["probabilities"]["home_win_regulation"] + result["probabilities"]["away_win_regulation"] + result["probabilities"]["draw_regulation"]
            assert total_prob == pytest.approx(1.0)

    @patch('src.mlsnp_predictor.advanced_predictor.MLPredictor')
    async def test_predict_match_ml(self, MockMLPredictor, mock_db_manager, sample_sim_data):
        # Arrange
        mock_ml_instance = MagicMock()
        mock_ml_instance.ml_model = MagicMock()
        mock_ml_instance._extract_features.return_value = {}
        mock_ml_instance.ml_model.predict.side_effect = [[1.5], [1.0]]
        type(mock_ml_instance).model_version = "test_model_v1"

        def mock_calc_win_prob(home_goals, away_goals):
            if home_goals > away_goals: return 0.6
            return 0.2
        mock_ml_instance._calculate_win_probability = MagicMock(side_effect=mock_calc_win_prob)

        MockMLPredictor.return_value = mock_ml_instance

        mock_db_manager.get_conference_teams.return_value = [{'id': 'team1', 'name': 'Team One'}, {'id': 'team2', 'name': 'Team Two'}]
        mock_db_manager.get_data_for_simulation.return_value = sample_sim_data

        with patch('src.mlsnp_predictor.advanced_predictor.SingleMatchPredictor._calculate_league_averages', new_callable=AsyncMock) as mock_avg:
            mock_avg.return_value = {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}
            predictor = SingleMatchPredictor(mock_db_manager)

            # Act
            result = await predictor.predict_match("team1", "team2", model_type='ml')

            # Assert
            assert "probabilities" in result
            assert "potential_score" in result
            assert result["model_version"] == "test_model_v1"
            assert result["probabilities"]["home_win_total"] == 0.6
            assert result["probabilities"]["away_win_total"] == 0.2
            assert result["probabilities"]["draw_regulation"] == pytest.approx(0.2)

@pytest.mark.asyncio
class TestEndSeasonScenarioGenerator:
    async def test_generate_scenarios_too_many_games(self, mock_db_manager):
        # Arrange
        mock_db_manager.get_conference_teams.return_value = [{'id': 'team1', 'name': 'Team One'}]
        sample_data = {
            "conference_teams": {"team1": "Team One"},
            "games_data": [{"is_completed": False, "home_team_id": "team1", "away_team_id": "team1"}] * 6,
            "team_performance": {}
        }
        mock_db_manager.get_data_for_simulation.return_value = sample_data
        generator = EndSeasonScenarioGenerator(mock_db_manager)

        # Act
        result = await generator.generate_scenarios("team1")

        # Assert
        assert "error" in result
        assert result["games_remaining"] == 6

    @patch('src.mlsnp_predictor.advanced_predictor.EndSeasonScenarioGenerator._get_game_probabilities')
    async def test_generate_scenarios_valid(self, mock_get_probs, mock_db_manager):
        # Arrange
        team_names = {"team1": "A", "team2": "B", "team3": "C", "team4": "D"}
        mock_db_manager.get_conference_teams.return_value = [{'id': k, 'name': v} for k,v in team_names.items()]

        games_data = [
            {"id": 1, "is_completed": False, "home_team_id": "team1", "away_team_id": "team2"},
            {"id": 2, "is_completed": False, "home_team_id": "team3", "away_team_id": "team4"},
        ]
        sample_data = {
            "conference_teams": team_names,
            "games_data": games_data,
            "team_performance": {t: {"games_played": 0} for t in team_names}
        }
        mock_db_manager.get_data_for_simulation.return_value = sample_data

        mock_get_probs.side_effect = [
            {"home_win_reg": 0.5, "away_win_reg": 0.3, "home_win_so": 0.1, "away_win_so": 0.1},
            {"home_win_reg": 0.4, "away_win_reg": 0.4, "home_win_so": 0.1, "away_win_so": 0.1},
        ]

        with patch('src.mlsnp_predictor.advanced_predictor.EndSeasonScenarioGenerator._calculate_league_averages', new_callable=AsyncMock) as mock_avg:
            mock_avg.return_value = {"league_avg_xgf": 1.2, "league_avg_xga": 1.2}

            with patch('src.mlsnp_predictor.advanced_predictor.MLPredictor') as MockMLPredictor:
                mock_ml_instance = MagicMock()
                mock_ml_instance.ml_model = True
                MockMLPredictor.return_value = mock_ml_instance

                generator = EndSeasonScenarioGenerator(mock_db_manager)

                # Act
                result = await generator.generate_scenarios("team1")

                # Assert
                assert "scenario_summary" in result
                total_prob = sum(rank_data['total_probability'] for rank_data in result['scenario_summary'])
                assert total_prob == pytest.approx(1.0)
                assert len(result["scenario_summary"]) > 1

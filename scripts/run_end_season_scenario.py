import asyncio
import os
import sys
import logging
from dotenv import load_dotenv
from databases import Database

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.database_manager import DatabaseManager
from src.mlsnp_predictor.advanced_predictor import EndSeasonScenarioGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main(team_id: str):
    """
    Generates and prints end-of-season scenarios for a given team.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL_PUBLIC")
    if not db_url:
        print("DATABASE_URL_PUBLIC environment variable not set.")
        return
    database = Database(db_url)
    await database.connect()

    try:
        db_manager = DatabaseManager(database)
        scenario_generator = EndSeasonScenarioGenerator(db_manager)
        scenarios = await scenario_generator.generate_scenarios(team_id)
        
        if "error" in scenarios:
            logger.error(f"Error generating scenarios: {scenarios['error']}")
            return

        logger.info(f"Scenarios for team {scenarios.get('target_team')}:")
        for summary in scenarios.get("scenario_summary", []):
            logger.info(f"  Final Rank: {summary['final_rank']}")
            logger.info(f"  Total Probability: {summary['total_probability']:.2%}")
            logger.info("  Example Scenarios:")
            for scenario in summary.get("example_scenarios", []):
                logger.info(f"    - Path: {', '.join(scenario['path'])}")
                logger.info(f"      Probability: {scenario['probability']:.4%}")

    finally:
        await database.disconnect()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_end_season_scenario.py <team_id>")
        sys.exit(1)
    
    team_id_arg = sys.argv[1]
    asyncio.run(main(team_id_arg))

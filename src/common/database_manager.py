"""
Database Manager for MLS Next Pro Predictor
This module handles all database operations, providing a unified interface
for storing and retrieving game data, team statistics, and simulation results.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from databases import Database
import asyncio
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Central database manager that handles all database operations.
    
    This class follows a "database-first" approach:
    1. Always check the database for existing data.
    2. Only fetch from ASA API if data is missing or outdated.
    3. Cache all fetched data in the database for future use.
    """
    
    def __init__(self, database: Database):
        """
        Initialize the database manager.
        
        Args:
            database: The database connection object from the 'databases' library.
        """
        self.db = database
        self._asa_client = None  # Lazy loaded only when needed

    async def initialize(self):
        """
        Initialize the database manager and ensure all foundational data,
        like conferences, exists in the database. This should be called
        after the database connection is established.
        """
        logger.info("Initializing database manager...")
        try:
            await self.ensure_conferences_exist()
            logger.info("Database manager initialization complete.")
        except Exception as e:
            logger.error(f"Error during DatabaseManager initialization: {e}", exc_info=True)
            raise
        
    @property
    def asa_client(self):
        """Lazy load ASA client only when needed"""
        if self._asa_client is None:
            from itscalledsoccer.client import AmericanSoccerAnalysis
            self._asa_client = AmericanSoccerAnalysis()
            logger.info("Initialized ASA client for data fetching.")
        return self._asa_client

    # ==================== Team & Conference Management ====================
    
    async def get_or_create_team(self, team_id: str, team_name: str, 
                                 team_abbv: str = None, conference_id: int = None) -> Dict:
        """
        Get a team from the database or create it if it doesn't exist.
        
        This method ensures consistent team data across the system.
        """
        try:
            query = "SELECT * FROM team WHERE team_id = :team_id"
            team = await self.db.fetch_one(query, values={"team_id": team_id})

            if team:
                return dict(team)

            logger.info(f"Team {team_id} not found, creating new team: {team_name}")

            insert_query = """
                INSERT INTO team (team_id, team_name, team_abbv, team_short_name, is_active, created_at)
                VALUES (:team_id, :team_name, :team_abbv, :short_name, true, NOW())
                RETURNING *
            """

            if not team_abbv:
                team_abbv = ''.join(word[0].upper() for word in team_name.split()[:3])

            short_name = team_name.split()[0] if team_name else team_abbv

            values = {
                "team_id": team_id,
                "team_name": team_name,
                "team_abbv": team_abbv,
                "short_name": short_name
            }

            new_team_record = await self.db.fetch_one(insert_query, values=values)

            if conference_id:
                await self.create_team_affiliation(team_id, conference_id)

            logger.info(f"Successfully created team: {team_name} ({team_id})")
            return dict(new_team_record)
        except Exception as e:
            logger.error(f"Database error in get_or_create_team for team_id {team_id}: {e}", exc_info=True)
            raise
    
    async def create_team_affiliation(self, team_id: str, conference_id: int, 
                                      league_id: int = 1, division_id: int = None):
        """
        Create or update a team's affiliation with a conference and league.
        This is crucial for tracking which teams belong to which conference over time.
        """
        try:
            check_query = """
                SELECT * FROM team_affiliations
                WHERE team_id = :team_id AND conference_id = :conference_id AND is_current = true
            """
            existing = await self.db.fetch_one(check_query, values={"team_id": team_id, "conference_id": conference_id})
            
            if existing:
                logger.debug(f"Team {team_id} already has current affiliation with conference {conference_id}.")
                return dict(existing)

            logger.info(f"Updating existing affiliations and creating new for team {team_id} with conference {conference_id}.")
            async with self.db.transaction(): # Ensure atomicity
                update_query = """
                    UPDATE team_affiliations SET is_current = false, end_date = CURRENT_DATE
                    WHERE team_id = :team_id AND is_current = true
                """
                await self.db.execute(update_query, values={"team_id": team_id})

                insert_query = """
                    INSERT INTO team_affiliations
                    (team_id, league_id, conference_id, division_id, start_date, is_current, created_at)
                    VALUES (:team_id, :league_id, :conference_id, :division_id, CURRENT_DATE, true, NOW())
                    RETURNING *
                """
                values = {
                    "team_id": team_id,
                    "league_id": league_id,
                    "conference_id": conference_id,
                    "division_id": division_id
                }

                affiliation = await self.db.fetch_one(insert_query, values=values)
            logger.info(f"Successfully created affiliation for team {team_id} with conference {conference_id}.")
            return dict(affiliation)
        except Exception as e:
            logger.error(f"Database error in create_team_affiliation for team_id {team_id}, conf_id {conference_id}: {e}", exc_info=True)
            raise

    async def get_conference_teams(self, conference_id: int, season_year: int) -> Dict[str, str]:
        """
        Get all teams for a given conference ID and season year.
        Returns a dictionary mapping team_id to team_name.
        
        Args:
            conference_id: 1 for Eastern, 2 for Western
            season_year: The season year to get teams for
        """
        from datetime import datetime, date
        
        logger.debug(f"Fetching teams for conference_id: {conference_id}, season: {season_year}")
        try:
            current_year = datetime.now().year
            
            if season_year == current_year:
                # For current year, use the is_current flag for efficiency
                query = """
                    SELECT t.team_id, t.team_name
                    FROM team t
                    JOIN team_affiliations ta ON t.team_id = ta.team_id
                    WHERE ta.conference_id = :conference_id
                    AND ta.is_current = true
                    AND t.is_active = true
                    ORDER BY t.team_name
                """
                values = {"conference_id": conference_id}
            else:
                # For historical years, check date ranges
                season_start = date(season_year, 1, 1)
                season_end = date(season_year, 12, 31)

                query = """
                    SELECT t.team_id, t.team_name
                    FROM team t
                    JOIN team_affiliations ta ON t.team_id = ta.team_id
                    WHERE ta.conference_id = :conference_id
                    AND t.is_active = true
                    AND (
                        (ta.start_date IS NULL OR ta.start_date <= :season_end) AND
                        (ta.end_date IS NULL OR ta.end_date >= :season_start)
                    )
                    ORDER BY t.team_name
                """
                values = {
                    "conference_id": conference_id,
                    "season_start": season_start,
                    "season_end": season_end
                }

            teams_records = await self.db.fetch_all(query, values=values)
            teams_dict = {team['team_id']: team['team_name'] for team in teams_records}
            logger.info(f"Found {len(teams_dict)} teams for conference_id: {conference_id}, season: {season_year}")
            return teams_dict
        except Exception as e:
            logger.error(f"Database error in get_conference_teams for conf_id {conference_id}, season {season_year}: {e}", exc_info=True)
            raise

    async def ensure_conferences_exist(self):
        """
        Ensures the Eastern and Western conferences exist in the database.
        This prevents errors in downstream processing that rely on conference data.
        """
        try:
            conferences = {1: 'Eastern Conference', 2: 'Western Conference'}
            for conf_id, conf_name in conferences.items():
                query = "SELECT * FROM conference WHERE conf_id = :conf_id"
                conference = await self.db.fetch_one(query, values={"conf_id": conf_id})
                if not conference:
                    logger.info(f"Conference {conf_name} (ID: {conf_id}) not found, creating.")
                    insert_query = """
                        INSERT INTO conference (conf_id, conf_name, league_id, created_at)
                        VALUES (:conf_id, :conf_name, 1, NOW())
                    """
                    await self.db.execute(insert_query, values={"conf_id": conf_id, "conf_name": conf_name})
                    logger.info(f"Inserted {conf_name} into database.")
                else:
                    logger.debug(f"Conference {conf_name} (ID: {conf_id}) already exists.")
        except Exception as e:
            logger.error(f"Database error in ensure_conferences_exist: {e}", exc_info=True)
            raise

    # ==================== Game Data Management ====================
    
    async def get_games_for_season(self, season_year: int, conference: str = None,
                                   include_incomplete: bool = False) -> List[Dict]:
        """
        Retrieve all games for a season from the database, with optional filters.
        """
        logger.debug(f"Fetching games for season: {season_year}, conference: {conference}, include_incomplete: {include_incomplete}")
        try:
            query = """
                SELECT g.*,
                       ht.team_name as home_team_name, at.team_name as away_team_name,
                       hta.conference_id as home_conference_id, ata.conference_id as away_conference_id
                FROM games g
                JOIN team ht ON g.home_team_id = ht.team_id
                JOIN team at ON g.away_team_id = at.team_id
                LEFT JOIN team_affiliations hta ON ht.team_id = hta.team_id AND hta.is_current = true
                LEFT JOIN team_affiliations ata ON at.team_id = ata.team_id AND ata.is_current = true
                WHERE g.season_year = :season_year AND g.status != 'postponed'
            """
            values = {"season_year": season_year}

            if conference:
                conf_id = 1 if conference.lower() == 'eastern' else 2
                query += " AND (hta.conference_id = :conf_id OR ata.conference_id = :conf_id)"
                values["conf_id"] = conf_id

            if not include_incomplete:
                query += " AND g.is_completed = true"

            query += " ORDER BY g.date, g.game_id"

            games_records = await self.db.fetch_all(query, values=values)
            games_list = [dict(game) for game in games_records]
            logger.info(f"Found {len(games_list)} games for season: {season_year}, conference: {conference}")
            return games_list
        except Exception as e:
            logger.error(f"Database error in get_games_for_season for year {season_year}, conf {conference}: {e}", exc_info=True)
            raise

    async def get_or_fetch_game(self, game_id: str) -> Optional[Dict]:
        """
        Get a single game from the database or fetch it from the API if missing.
        """
        game: Optional[Dict] = None  # Initialize game
        try:
            query = "SELECT * FROM games WHERE game_id = :game_id"
            game = await self.db.fetch_one(query, values={"game_id": game_id})

            if not game:
                logger.warning(f"Game {game_id} not found in database")
                return None

            # Proceed only if game was found    
            if game['asa_loaded']:
                return dict(game)
        
            # Extract team IDs and date from the existing game record
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']
            game_date = game['date']
            season_year = game['season_year']
            
            logger.info(f"Game {game_id} needs ASA data, fetching by teams and date...")
            try:
                # Fetch games for both teams for the season
                games_data = self.asa_client.get_games(
                    leagues=['mlsnp'],
                    team_ids=[home_team_id, away_team_id],
                    seasons=[str(season_year)]  # Changed from season_name
                )
                
                if games_data.empty:
                    logger.warning(f"No ASA games found for teams {home_team_id} vs {away_team_id} in {season_year}")
                    return dict(game)
                
                # Convert game_date to date object for comparison
                target_date = game_date.date() if hasattr(game_date, 'date') else game_date
                
                # Find the matching game by home/away teams and date
                for idx, asa_game in games_data.iterrows():
                    # Parse ASA date
                    asa_date_str = asa_game.get('date_time_utc', asa_game.get('date'))
                    if asa_date_str:
                        try:
                            # Use pandas to parse and convert to timezone-naive
                            asa_datetime = pd.to_datetime(asa_date_str)
                            asa_date = asa_datetime.tz_localize(None).date() if asa_datetime.tzinfo else asa_datetime.date()
                        except Exception as e:
                            logger.warning(f"Could not parse ASA date: {asa_date_str}, error: {e}")
                            continue
                        
                        # Check if this is our game (same teams and within 1 day)
                        date_diff = abs((asa_date - target_date).days)
                        if (asa_game['home_team_id'] == home_team_id and 
                            asa_game['away_team_id'] == away_team_id and
                            date_diff <= 1):  # Allow 1 day difference for timezone issues
                            
                            # Found the matching game!
                            game_data = asa_game.to_dict()
                            
                            # IMPORTANT: Preserve our original game_id
                            game_data['game_id'] = game_id
                            
                            # Store with ASA data
                            await self.store_game(game_data, from_asa=True)
                            
                            # Return the updated record
                            updated_game = await self.db.fetch_one(query, values={"game_id": game_id})
                            return dict(updated_game) if updated_game else None
                
                logger.warning(f"Could not find exact match in ASA data for {home_team_id} vs {away_team_id} on {target_date}")
                return dict(game)
                
            except Exception as e:
                logger.error(f"Error fetching game {game_id} from ASA: {e}", exc_info=True)
                # If game was fetched from DB but ASA fetch failed, return the DB game.
                # If game is None (e.g. initial DB fetch failed and was caught by outer except), this will be None.
                return dict(game) if game else None
        except Exception as e: # This is the missing outer except block
            logger.error(f"Error in get_or_fetch_game for game {game_id}: {e}", exc_info=True)
            return None # Return None if any error occurs in the outer try block
        
    def _convert_nan_to_none_scalar(self, value):
        """Converts a single NaN value (float('nan') or np.nan) to None."""
        if isinstance(value, float) and np.isnan(value):
            return None
        # Handle numpy specific NaN types (e.g., from pandas Series.to_dict())
        if isinstance(value, np.generic) and np.isscalar(value) and np.isnan(value):
            return None
        return value    
    
    async def store_game(self, game_data: Dict, from_asa: bool = False) -> Dict:
        """
        Store a game in the database, handling data transformation and updates.
        Uses ON CONFLICT to efficiently update existing game records.
        Now also updates matchdays for existing games when ASA data is provided.
        """
        try:
            # Helper function to convert pandas/numpy values to Python bool
            def to_bool(value):
                """Convert various representations to boolean."""
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return False
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                if isinstance(value, (int, float)):
                    return bool(value) and value != 0
                if isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes', 't', 'y']
                return False
            
            # Handle scores - ensure they're integers or None
            def safe_int(value):
                if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
                    return None
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None

            game_id = game_data.get('game_id')
            if not game_id:
                raise ValueError("game_id is required")
            
            # Handle date parsing
            date_str = game_data.get('date_time_utc', game_data.get('date'))
            game_date = None
            
            if date_str:
                if isinstance(date_str, datetime):
                    game_date = date_str.replace(tzinfo=None) if date_str.tzinfo else date_str
                elif isinstance(date_str, pd.Timestamp):
                    game_date = date_str.to_pydatetime().replace(tzinfo=None)
                elif isinstance(date_str, str):
                    # Clean the string and parse
                    date_str_clean = date_str.replace(' UTC', '').replace('Z', '')
                    try:
                        parsed_date = pd.to_datetime(date_str_clean, utc=True)
                        game_date = parsed_date.tz_localize(None)
                    except:
                        # Fallback parsing
                        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'):
                            try:
                                game_date = datetime.strptime(date_str_clean, fmt)
                                break
                            except ValueError:
                                continue

            if not game_date:
                raise ValueError(f"Could not parse date: {date_str}")
                
            home_score = safe_int(game_data.get('home_score'))
            away_score = safe_int(game_data.get('away_score'))
            home_penalties = safe_int(game_data.get('home_penalties', 0))
            away_penalties = safe_int(game_data.get('away_penalties', 0))
            original_asa_data = None
            correction_notes = None
            data_corrected = False
            
            # Handle shootout logic - use to_bool
            went_to_shootout = to_bool(game_data.get('penalties', game_data.get('went_to_shootout', False)))
            
            # Handle status and completion - use to_bool for boolean fields
            status = game_data.get('status', 'scheduled').lower()
            
            if from_asa:
                # Store original ASA data before any corrections
                original_asa_data_dict_raw = {
                    'home_score': self._convert_nan_to_none_scalar(game_data.get('home_score')),
                    'away_score': self._convert_nan_to_none_scalar(game_data.get('away_score')),
                    'penalties': self._convert_nan_to_none_scalar(game_data.get('penalties')),
                    'home_penalties': self._convert_nan_to_none_scalar(game_data.get('home_penalties')),
                    'away_penalties': self._convert_nan_to_none_scalar(game_data.get('away_penalties')),
                    # For went_to_shootout_original, get the raw value (which might be NaN) and convert to None if so
                    'went_to_shootout_original': self._convert_nan_to_none_scalar(game_data.get('penalties', game_data.get('went_to_shootout', False)))
                }
                original_asa_data = json.dumps(original_asa_data_dict_raw) # Use the dict with NaNs converted to None

                # For ASA data: determine completion based on actual score data
                # ASA should only return games that have been played
                has_scores = (home_score is not None and away_score is not None)
                is_completed = has_scores
                
                # Fix invalid shootouts
                if went_to_shootout and home_score is not None and away_score is not None:
                    if home_score != away_score:
                        logger.warning(f"Game {game_id}: ASA marked as shootout but score is {home_score}-{away_score}. Auto-correcting.")
                        went_to_shootout = False
                        home_penalties = None
                        away_penalties = None
                        data_corrected = True
                        correction_notes = f"Auto-corrected: Game marked as shootout but regulation score was {home_score}-{away_score}"

                # ADDITIONAL: Handle missing penalty data for shootouts
                if went_to_shootout and (home_penalties is None or away_penalties is None):
                    if not data_corrected:  # Don't overwrite previous correction
                        data_corrected = True
                        correction_notes = f"Auto-corrected: Shootout game missing penalty data"
                    else:
                        correction_notes += "; Also fixed missing penalty data"
                    logger.warning(f"Game {game_id}: Shootout game missing penalty data, setting to 0-0")
                    home_penalties = 0
                    away_penalties = 0
            else:
                is_completed = to_bool(game_data.get('is_completed', False))
            

            # Log a warning for data quality issues
            if is_completed and (home_score is None or away_score is None) and not went_to_shootout:
                logger.warning(f"Game {game_id} is completed but has no final score.")
            
            # Handle integer fields safely
            matchday = safe_int(game_data.get('matchday', 0)) or 0
            attendance = safe_int(game_data.get('attendance', 0)) or 0
            expanded_minutes = safe_int(game_data.get('expanded_minutes', 90)) or 90
            season_year = safe_int(game_data.get('season_year', game_date.year)) or game_date.year
            
            values = {
                "game_id": game_id,
                "date": game_date,
                "status": status,
                "home_team_id": game_data.get('home_team_id'),
                "away_team_id": game_data.get('away_team_id'),
                "matchday": matchday,
                "attendance": attendance,
                "is_completed": is_completed,
                "went_to_shootout": went_to_shootout,
                "season_year": season_year,
                "expanded_minutes": expanded_minutes,
                "home_score": home_score if is_completed else None,
                "away_score": away_score if is_completed else None,
                "home_penalties": home_penalties if went_to_shootout else None,
                "away_penalties": away_penalties if went_to_shootout else None,
                "asa_loaded": bool(from_asa),
                "data_corrected": data_corrected,
                "correction_notes": correction_notes,
                "original_asa_data": original_asa_data
            }
            
            # Debug logging
            if from_asa and is_completed:
                logger.info(f"Storing completed ASA game {game_id}: {values['home_team_id']} {home_score}-{away_score} {values['away_team_id']}")
            
            query = """
                INSERT INTO games (
                    game_id, date, status, home_team_id, away_team_id, matchday,
                    attendance, is_completed, went_to_shootout, season_year, 
                    expanded_minutes, home_score, away_score, home_penalties, 
                    away_penalties, asa_loaded, data_corrected, correction_notes, 
                    original_asa_data, created_at, updated_at)
                VALUES (
                    :game_id, :date, :status, :home_team_id, :away_team_id, :matchday,
                    :attendance, :is_completed, :went_to_shootout, :season_year,
                    :expanded_minutes, :home_score, :away_score, :home_penalties,
                    :away_penalties, :asa_loaded, :data_corrected, :correction_notes,
                    :original_asa_data, NOW(), NOW())
                ON CONFLICT (game_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    is_completed = EXCLUDED.is_completed,
                    went_to_shootout = EXCLUDED.went_to_shootout,
                    matchday = CASE WHEN EXCLUDED.matchday > 0 THEN EXCLUDED.matchday ELSE games.matchday END,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    home_penalties = EXCLUDED.home_penalties,
                    away_penalties = EXCLUDED.away_penalties,
                    attendance = EXCLUDED.attendance,
                    expanded_minutes = EXCLUDED.expanded_minutes,
                    asa_loaded = games.asa_loaded OR EXCLUDED.asa_loaded,
                    data_corrected = games.data_corrected OR EXCLUDED.data_corrected,
                    correction_notes = COALESCE(EXCLUDED.correction_notes, games.correction_notes),
                    original_asa_data = COALESCE(EXCLUDED.original_asa_data, games.original_asa_data),
                    updated_at = NOW()
                RETURNING *
            """
            
            result = await self.db.fetch_one(query, values=values)
            return dict(result)
        
        except Exception as e:
            logger.error(f"Error in store_game for game_id {game_data.get('game_id', 'unknown')}: {e}", exc_info=True)
            raise

    async def update_games_with_asa(self, season_year: int, conference: str = None):
        """
        Finds and updates all games that have been played, but ASA data is missing, or
        should have been played but are still marked as incomplete in the database and
        also need ASA data.
        """
        # Create timezone-naive datetime to match database storage
        cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)  # 2 hour buffer

        if conference:
            conf_id = 1 if conference.lower() == 'eastern' else 2
            query = """
                SELECT g.game_id, g.home_team_id, g.away_team_id, g.date, g.is_completed
                FROM games g
                JOIN team_affiliations hta ON g.home_team_id = hta.team_id AND hta.is_current = true
                JOIN team_affiliations ata ON g.away_team_id = ata.team_id AND ata.is_current = true
                WHERE g.season_year = :season_year 
                AND (g.asa_loaded = false OR (g.is_completed = false AND g.date < :cutoff_date))
                AND (hta.conference_id = :conf_id OR ata.conference_id = :conf_id)
                ORDER BY g.date
            """
            values = {
                "season_year": season_year, 
                "cutoff_date": cutoff_date,
                "conf_id": conf_id
            }
        else:
            query = """
                SELECT game_id, home_team_id, away_team_id, date, is_completed 
                FROM games 
                WHERE season_year = :season_year 
                AND (asa_loaded = false OR (is_completed = false AND date < :cutoff_date))
                ORDER BY date
            """
            values = {
                "season_year": season_year, 
                "cutoff_date": cutoff_date
            }
        
        games_to_update = await self.db.fetch_all(query, values=values)
        
        if not games_to_update:
            logger.info(f"All games for {season_year} are up to date with ASA data.")
            return

        logger.info(f"Found {len(games_to_update)} games to update with ASA data for {season_year}.")
        
        # Get unique team IDs to limit ASA fetch
        team_ids = set()
        for game in games_to_update:
            team_ids.add(game['home_team_id'])
            team_ids.add(game['away_team_id'])

        # Fetch all ASA games for the season once
        try:
            logger.info(f"Fetching ASA games for {len(team_ids)} teams...")
            # Fetch only games involving these specific teams
            all_asa_games = self.asa_client.get_games(
                leagues=['mlsnp'],
                team_ids=list(team_ids),  # Limit to only relevant teams
                seasons=[str(season_year)]
            )
            
            if all_asa_games.empty:
                logger.warning(f"No games found in ASA for the specified teams in {season_year}")
                return
                
            logger.info(f"Found {len(all_asa_games)} games in ASA for {season_year}")
            
            # Create a lookup dictionary for ASA games
            asa_lookup = {}
            for idx, asa_game in all_asa_games.iterrows():
                # Parse ASA date with better error handling
                asa_date_str = asa_game.get('date_time_utc', asa_game.get('date'))
                if asa_date_str:
                    try:
                        # Handle different date formats from ASA
                        if isinstance(asa_date_str, str):
                            # Remove timezone info for consistent parsing
                            clean_date_str = asa_date_str.replace(' UTC', '').replace('Z', '')
                            asa_datetime = pd.to_datetime(clean_date_str, utc=True)
                        else:
                            asa_datetime = pd.to_datetime(asa_date_str, utc=True)
                        
                        # Convert to date for matching (removes time component)
                        asa_date = asa_datetime.date()
                        
                        # Create lookup key
                        key = (asa_game['home_team_id'], asa_game['away_team_id'], asa_date)
                        asa_lookup[key] = asa_game.to_dict()
                        
                        # ADDED: Also create keys for adjacent dates to handle timezone issues
                        for day_offset in [-1, 1]:
                            alt_date = asa_date + timedelta(days=day_offset)
                            alt_key = (asa_game['home_team_id'], asa_game['away_team_id'], alt_date)
                            if alt_key not in asa_lookup:  # Don't overwrite exact matches
                                asa_lookup[alt_key] = asa_game.to_dict()
                                
                    except Exception as e:
                        logger.warning(f"Could not parse ASA date: {asa_date_str}, error: {e}")
                        continue
            
            logger.info(f"Created lookup with {len(asa_lookup)} date/team combinations")
            
            # Now update each game that needs ASA data
            updated_count = 0
            no_match_count = 0
            
            for game_record in games_to_update:
                game_id = game_record['game_id']

                game_date = game_record['date'].date() if hasattr(game_record['date'], 'date') else game_record['date']
                lookup_key = (game_record['home_team_id'], game_record['away_team_id'], game_date)
                
                if lookup_key in asa_lookup:
                    asa_data = asa_lookup[lookup_key]
                    # CRITICAL: Preserve our game_id
                    asa_data['game_id'] = game_id
                    
                    await self.store_game(asa_data, from_asa=True)
                    updated_count += 1
                    
                    if updated_count % 5 == 0:
                        logger.info(f"Updated {updated_count} games with ASA data...")
                else:
                    no_match_count += 1
                    logger.debug(f"No ASA match found for game {game_id}: {lookup_key}")
            
            logger.info(f"Successfully updated {updated_count} games with ASA data. {no_match_count} games had no match.")
            
        except Exception as e:
            logger.error(f"Error in bulk ASA update: {e}", exc_info=True)
            raise

    async def get_corrected_games_summary(self, season_year: int = 2025) -> Dict:
        """Get summary of all data corrections made."""
        query = """
            SELECT 
                COUNT(*) as total_corrected,
                correction_notes,
                COUNT(*) as count_by_type
            FROM games 
            WHERE data_corrected = true 
            AND season_year = :season_year
            GROUP BY correction_notes
            ORDER BY count_by_type DESC
        """
        
        corrections = await self.db.fetch_all(query, values={"season_year": season_year})
        
        return {
            "total_corrections": sum(c['count_by_type'] for c in corrections),
            "correction_types": [dict(c) for c in corrections]
        }


    # ==================== Team Statistics Management ====================
    
    async def get_or_fetch_team_xg(self, team_id: str, season_year: int) -> Optional[Dict]:
        """
        Get team expected goals (xG) stats from the database or fetch from the API.
        Data is considered stale after 1 day and will be re-fetched.
        
        UPDATED: Removed x_shots_for/x_shots_against fields (not available in ASA MLSNP data)
        """
        query = """
            SELECT * FROM team_xg_history 
            WHERE team_id = :team_id AND season_year = :season_year
        """
        xg_data = await self.db.fetch_one(query, values={"team_id": team_id, "season_year": season_year})

        if xg_data and (datetime.now(timezone.utc) - xg_data['date_captured']) < timedelta(days=1):
            logger.debug(f"Returning fresh xG data for team {team_id} from database.")
            return dict(xg_data)

        logger.info(f"Fetching new xG data for team {team_id}, season {season_year} from ASA.")
        try:
            stats_df = self.asa_client.get_team_xgoals(
                leagues=['mlsnp'],
                team_ids=[team_id], 
                season_name=[str(season_year)]
            )
            
            if stats_df.empty:
                logger.warning(f"No xG data returned from ASA API for team {team_id}, season {season_year}.")
                return None

            stat = stats_df.iloc[0].to_dict()
            
            insert_values = {
                "team_id": team_id,
                "games_played": stat.get('count_games', 0),
                "shots_for": stat.get('shots_for', 0),
                "shots_against": stat.get('shots_against', 0),
                "x_goals_for": stat.get('xgoals_for', 0.0),
                "x_goals_against": stat.get('xgoals_against', 0.0),
                "season_year": season_year
            }
            
            insert_query = """
                INSERT INTO team_xg_history (
                    team_id, season_year, games_played, shots_for, shots_against, 
                    x_goals_for, x_goals_against, date_captured
                ) VALUES (
                    :team_id, :season_year, :games_played, :shots_for, :shots_against,
                    :x_goals_for, :x_goals_against, NOW()
                )
                ON CONFLICT (team_id, season_year) DO UPDATE SET
                    games_played = EXCLUDED.games_played,
                    shots_for = EXCLUDED.shots_for,
                    shots_against = EXCLUDED.shots_against,
                    x_goals_for = EXCLUDED.x_goals_for,
                    x_goals_against = EXCLUDED.x_goals_against,
                    date_captured = NOW()
                RETURNING *
            """
            
            stored_data = await self.db.fetch_one(insert_query, values=insert_values)
            logger.info(f"Successfully stored xG data for team {team_id}, season {season_year}.")
            return dict(stored_data)
        except Exception as e:
            logger.error(f"Error fetching xG data for team {team_id}: {e}", exc_info=True)
            return None

    # ==================== Standings Management ====================
    
    async def calculate_and_store_standings(self, season_year: int, conference: str) -> List[Dict]:
        """
        Calculates current standings from completed games and stores them.
        Correctly implements the MLS Next Pro point system, including detailed goal stats.
        """
        conf_id = 1 if conference.lower() == 'eastern' else 2
        conf_teams = await self.get_conference_teams(conf_id, season_year)
        games = await self.get_games_for_season(season_year, conference, include_incomplete=False)

        standings = defaultdict(lambda: {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,  # Games that went to shootout
            "so_wins": 0,  # Shootout wins
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
            "home_goals_for": 0,
            "home_goals_against": 0,
            "away_goals_for": 0,
            "away_goals_against": 0
        })

        for team_id in conf_teams:
            standings[team_id] # Initialize every team in the conference

        for game in games:
            home_id, away_id = game['home_team_id'], game['away_team_id']
            home_score, away_score = game['home_score'], game['away_score']

            if home_id not in conf_teams or away_id not in conf_teams:
                continue # Skip games not between two conference teams

            # Increment games played for both teams
            standings[home_id]["games_played"] += 1
            standings[away_id]["games_played"] += 1

            # Update overall goals for and against
            standings[home_id]["goals_for"] += home_score
            standings[home_id]["goals_against"] += away_score
            standings[away_id]["goals_for"] += away_score
            standings[away_id]["goals_against"] += home_score

            # Update overall goal difference
            standings[home_id]["goal_difference"] += home_score - away_score
            standings[away_id]["goal_difference"] += away_score - home_score

            # Update home/away specific goals
            standings[home_id]["home_goals_for"] += home_score
            standings[home_id]["home_goals_against"] += away_score
            standings[away_id]["away_goals_for"] += away_score
            standings[away_id]["away_goals_against"] += home_score

            # Award points and update specific win/loss/draw stats
            if game['went_to_shootout']:
                standings[home_id]["draws"] += 1
                standings[away_id]["draws"] += 1
                if game['home_penalties'] > game['away_penalties']:
                    standings[home_id]["points"] += 2
                    standings[home_id]["so_wins"] += 1
                    standings[away_id]["points"] += 1
                    standings[away_id]["losses"] += 1 # A shootout loss is still a loss in this context
                else:
                    standings[away_id]["points"] += 2
                    standings[away_id]["so_wins"] += 1
                    standings[home_id]["points"] += 1
                    standings[home_id]["losses"] += 1 # A shootout loss is still a loss in this context
            else: # Regulation result
                if home_score > away_score:
                    standings[home_id]["points"] += 3
                    standings[home_id]["wins"] += 1
                    standings[away_id]["losses"] += 1
                elif away_score > home_score:
                    standings[away_id]["points"] += 3
                    standings[away_id]["wins"] += 1
                    standings[home_id]["losses"] += 1
                else:
                    # This case should ideally not happen for MLSNP games that go to shootout.
                    # If it does, it implies a regulation draw that didn't go to shootout,
                    # which would be 1 point for each, but MLSNP always goes to shootout.
                    # Keeping it for robustness, assuming it defaults to 1 point each like a regular draw.
                    standings[home_id]["draws"] += 1
                    standings[away_id]["draws"] += 1
                    standings[home_id]["points"] += 1
                    standings[away_id]["points"] += 1

        standings_list = [{"team_id": tid, **stats} for tid, stats in standings.items()]

        # Sort by MLS Next Pro tiebreakers
        # The tiebreakers are: Points, Wins, Goal Difference, Goals For, Shootout Wins
        # Assuming lower rank is better, so 'reverse=True' for sorting largest values first for points, wins, GD, GF, SO Wins.
        standings_list.sort(key=lambda x: (
            -x["points"],
            -x["wins"],
            -x["goal_difference"],
            -x["goals_for"],
            -x["so_wins"]
        )) # reverse=False by default, so sorting descending already

        return standings_list
    
    async def store_standings_record(self, team_id: str, season_year: int, matchday: int, stats: Dict) -> Dict:
        """
        Stores or updates a single team's standings record for a given season and matchday.
        Uses an UPSERT statement to handle conflicts.
        """
        insert_query = """
            INSERT INTO standings_history (
                team_id, season_year, matchday, games_played, wins, losses, 
                draws, so_wins, goals_for, goals_against, goal_difference, 
                points, home_goals_for, home_goals_against, away_goals_for, 
                away_goals_against, date_captured
            ) VALUES (
                :team_id, :season_year, :matchday, :games_played, :wins, :losses,
                :draws, :so_wins, :goals_for, :goals_against, :goal_difference,
                :points, :home_goals_for, :home_goals_against, :away_goals_for,
                :away_goals_against, NOW()
            )
            ON CONFLICT (team_id, season_year, matchday) DO UPDATE SET
                games_played = EXCLUDED.games_played,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                draws = EXCLUDED.draws,
                so_wins = EXCLUDED.so_wins,
                goals_for = EXCLUDED.goals_for,
                goals_against = EXCLUDED.goals_against,
                goal_difference = EXCLUDED.goal_difference,
                points = EXCLUDED.points,
                home_goals_for = EXCLUDED.home_goals_for,
                home_goals_against = EXCLUDED.home_goals_against,
                away_goals_for = EXCLUDED.away_goals_for,
                away_goals_against = EXCLUDED.away_goals_against,
                date_captured = NOW()
            RETURNING *
        """

        values = {
            "team_id": team_id,
            "season_year": season_year,
            "matchday": matchday,
            **stats # Unpack all the stats from the provided dictionary
        }
        
        try:
            result = await self.db.fetch_one(insert_query, values=values)
            logger.debug(f"Stored standings for team {team_id} (Season: {season_year}, Matchday: {matchday})")
            return dict(result)
        except Exception as e:
            logger.error(f"Error storing standings for team {team_id}: {e}", exc_info=True)
            raise

    
    # ==================== Fixture Management ====================
    
    async def store_fixtures(self, fixtures: List[Dict]):
        """
        Store scraped fixtures in the database as future games.
        """
        for fixture in fixtures:
            # Parse date
            fixture_date = None
            if fixture.get('date'):
                try:
                    fixture_date = datetime.strptime(fixture['date'], '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Could not parse fixture date: {fixture['date']}")
                    continue
            
            # Generate a unique game_id for fixtures
            game_id = f"{fixture['home_team_id']}_{fixture['away_team_id']}_{fixture['date']}"
            
            # Ensure both teams exist in the database
            if fixture.get('home_team_name'):
                await self.get_or_create_team(
                    fixture['home_team_id'], 
                    fixture['home_team_name']
                )
            if fixture.get('away_team_name'):
                await self.get_or_create_team(
                    fixture['away_team_id'], 
                    fixture['away_team_name']
                )
            
            # Create game record
            values = {
                "game_id": game_id,
                "date": fixture_date,
                "status": "scheduled",
                "home_team_id": fixture['home_team_id'],
                "away_team_id": fixture['away_team_id'],
                "matchday": fixture.get('matchday', 0),
                "attendance": 0,
                "is_completed": False,
                "went_to_shootout": False,
                "season_year": fixture_date.year if fixture_date else 2025,
                "expanded_minutes": 90,
                "home_score": None,
                "away_score": None,
                "home_penalties": None,
                "away_penalties": None
            }
            
            query = """
                INSERT INTO games (
                    game_id, date, status, home_team_id, away_team_id, matchday,
                    attendance, is_completed, went_to_shootout, season_year,
                     expanded_minutes, home_score, away_score, home_penalties,
                     away_penalties, created_at, updated_at
                ) VALUES (
                    :game_id, :date, :status, :home_team_id, :away_team_id, :matchday,
                    :attendance, :is_completed, :went_to_shootout, :season_year,
                    :expanded_minutes, :home_score, :away_score, :home_penalties,
                    :away_penalties, NOW(), NOW()
                )
                ON CONFLICT (game_id) DO NOTHING
            """
            
        try:
            await self.db.execute(query, values=values)
            logger.debug(f"Stored fixture: {game_id}")
        except Exception as e:
            logger.error(f"Error storing fixture {game_id}: {e}")

    # ==================== Simulation & Historical Data ====================

    async def get_data_for_simulation(self, conference: str, season_year: int) -> Dict[str, Any]:
        """
        Gathers all necessary data for a regular season simulation run.
        """
        logger.info(f"Gathering simulation data for {conference} conference, {season_year} season...")
        
        conf_id = 1 if conference.lower() == 'eastern' else 2
        conference_teams = await self.get_conference_teams(conf_id, season_year)
        team_ids = list(conference_teams.keys())
        
        await self.update_games_with_asa(season_year)
        
        all_games = await self.get_games_for_season(season_year, conference, include_incomplete=True)
        
        team_performance = {}
        for team_id in team_ids:
            xg_data = await self.get_or_fetch_team_xg(team_id, season_year)
            if xg_data and xg_data.get('games_played', 0) > 0:
                team_performance[team_id] = xg_data
            else: # Fallback for teams with no xG data
                team_performance[team_id] = {"team_id": team_id, "x_goals_for": 1.0, "x_goals_against": 1.0, "games_played": 0}

        logger.info(f"Successfully gathered data for {len(conference_teams)} teams.")
        return {
            "conference_teams": conference_teams,
            "games_data": all_games,
            "team_performance": team_performance
        }

    async def load_historical_season(self, season_year: int):
        """
        Loads all games and statistics for a historical season from the ASA API
        into the local database.
        """
        logger.info(f"Loading historical data for {season_year} season...")
        try:
            games_df = self.asa_client.get_games(leagues=['mlsnp'], season_name=[str(season_year)])
            logger.info(f"Found {len(games_df)} games for {season_year} season.")
            
            # Corrected iteration over DataFrame rows
            for _, game_row in games_df.iterrows():
                await self.store_game(game_row.to_dict(), from_asa=True)
                if len(games_df) > 100: # Still use original length for sleep logic
                    await asyncio.sleep(0.01)

            team_ids = await self.db.fetch_all("SELECT DISTINCT team_id FROM team")
            for team in team_ids:
                await self.get_or_fetch_team_xg(team['team_id'], season_year)
                await asyncio.sleep(0.1)
            
            logger.info(f"Successfully loaded and processed {season_year} season data.")
        except Exception as e:
            logger.error(f"Error loading historical season {season_year}: {e}", exc_info=True)
            raise

    async def update_matchday_from_asa_data(self, asa_game_data: Dict) -> bool:
        """
        Updates the matchday for existing games in the database using ASA game data.
        Matches games by home_team_id, away_team_id, and date since game_ids may differ.
        
        Args:
            asa_game_data: Game data from ASA API
            
        Returns:
            bool: True if a matching game was found and updated, False otherwise
        """
        try:
            # Extract the necessary data from ASA game
            home_team_id = asa_game_data.get('home_team_id')
            away_team_id = asa_game_data.get('away_team_id')
            matchday = asa_game_data.get('matchday')
            
            # Parse the date from ASA data
            date_str = asa_game_data.get('date_time_utc')
            if not date_str:
                logger.warning("No date found in ASA game data")
                return False
                
            # Parse the date to match our database format
            try:
                if isinstance(date_str, str):
                    # Handle the UTC format from ASA: "2025-05-24 23:00:00 UTC"
                    date_str = date_str.replace(' UTC', '+00:00')
                    for fmt in ('%Y-%m-%d %H:%M:%S%z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d'):
                        try:
                            game_date = datetime.strptime(date_str, fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        logger.warning(f"Could not parse ASA date: {date_str}")
                        return False
                else:
                    game_date = date_str.date() if hasattr(date_str, 'date') else date_str
            except Exception as e:
                logger.warning(f"Error parsing date from ASA data: {e}")
                return False
            
            if not all([home_team_id, away_team_id, game_date]):
                logger.warning("Missing required data for game matching")
                return False
                
            # Find existing games that match by teams and date
            query = """
                SELECT game_id, matchday FROM games 
                WHERE home_team_id = :home_team_id 
                AND away_team_id = :away_team_id 
                AND DATE(date) = :game_date
            """
            
            existing_games = await self.db.fetch_all(
                query, 
                values={
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id, 
                    "game_date": game_date
                }
            )
            
            if not existing_games:
                logger.debug(f"No existing games found for {home_team_id} vs {away_team_id} on {game_date}")
                return False
                
            # Update matchday for all matching games (should typically be just one)
            updated_count = 0
            for existing_game in existing_games:
                # Only update if we have valid matchday data and it's different
                if matchday is not None and matchday != existing_game['matchday']:
                    update_query = """
                        UPDATE games 
                        SET matchday = :matchday, updated_at = NOW()
                        WHERE game_id = :game_id
                    """
                    
                    await self.db.execute(
                        update_query,
                        values={
                            "matchday": int(matchday),
                            "game_id": existing_game['game_id']
                        }
                    )
                    
                    logger.info(f"Updated matchday to {matchday} for game {existing_game['game_id']}")
                    updated_count += 1
                
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Error updating matchday from ASA data: {e}", exc_info=True)
            return False
        
    async def get_team_goals_added_data(self, team_id: str, season_year: int) -> Dict:
        """
        Get Goals Added data for a team for a specific season.
        Aggregates across all action types to get total g+ for/against.
        """
        query = """
            SELECT 
                team_id,
                season_year,
                SUM(goals_added_for) as total_goals_added_for,
                SUM(goals_added_against) as total_goals_added_against,
                SUM(num_actions_for) as total_actions_for,
                SUM(num_actions_against) as total_actions_against,
                MAX(minutes_played) as minutes_played,
                COUNT(DISTINCT action_type) as action_types_count
            FROM team_ga_history 
            WHERE team_id = :team_id 
            AND season_year = :season_year
            GROUP BY team_id, season_year
        """
        
        result = await self.db.fetch_one(
            query, 
            values={"team_id": team_id, "season_year": season_year}
        )
        
        if result:
            # Estimate games played from minutes (assuming 90 min per game)
            estimated_games = max(1, result['minutes_played'] // 90) if result['minutes_played'] else 1
            
            return {
                'team_id': result['team_id'],
                'season_year': result['season_year'],
                'total_goals_added_for': result['total_goals_added_for'] or 0,
                'total_goals_added_against': result['total_goals_added_against'] or 0,
                'games_played': estimated_games,
                'action_types_available': result['action_types_count']
            }
        
        return None

    # ==================== Simulation Result Storage ====================

    def _safe_numeric(self, value, default=None, as_int=False):
        """
        Safely convert values to numeric, handling NaN, None, 'N/A', etc.
        
        Args:
            value: The value to convert
            default: Default value if conversion fails
            as_int: If True, convert to integer
        
        Returns:
            Converted numeric value or default
        """
        if pd.isna(value) or value == 'N/A' or value is None:
            return default
        try:
            result = float(value)
            return int(result) if as_int else result
        except (ValueError, TypeError):
            return default

    async def store_simulation_run(self, user_id: int, conference: str, n_simulations: int, season_year: int) -> int:
        """
        Stores a new simulation run record and returns the run ID.
        """
        logger.info(f"Storing simulation run: UserID {user_id}, Conf: {conference}, N_sim: {n_simulations}, Year: {season_year}")
        try:
            # Start a transaction since we need to insert into two tables
            async with self.db.transaction():
                # First, insert into prediction_runs
                run_query = """
                    INSERT INTO prediction_runs (
                        conference_id, n_simulations, season_year, run_date, 
                        is_stored, matchday, league_id
                    ) VALUES (
                        :conference_id, :n_simulations, :season_year, NOW(), 
                        TRUE, :matchday, :league_id
                    )
                    RETURNING run_id
                """
                
                # Map conference name to ID
                conf_map = {"eastern": 1, "western": 2}
                conference_id = conf_map.get(conference.lower())
                
                if conference_id is None:
                    # Handle "both" - might need to run two separate simulations
                    raise ValueError(f"Invalid conference: {conference}. Use 'eastern' or 'western'")
                
                # Get current matchday (you might want to calculate this dynamically)
                # For now, using a placeholder
                current_matchday = await self.get_current_matchday(season_year)
                
                run_values = {
                    "conference_id": conference_id,
                    "n_simulations": n_simulations,
                    "season_year": season_year,
                    "matchday": current_matchday or 0,
                    "league_id": 1  # MLS Next Pro
                }
                
                run_result = await self.db.fetch_one(run_query, values=run_values)
                run_id = run_result['run_id']
                
                # Second, link this run to the user
                user_sim_query = """
                    INSERT INTO user_simulations (user_id, run_id, created_at)
                    VALUES (:user_id, :run_id, NOW())
                """
                
                await self.db.execute(user_sim_query, values={
                    "user_id": user_id,
                    "run_id": run_id
                })
                
                logger.info(f"Stored new prediction run with ID: {run_id} for user {user_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Database error in store_simulation_run: {e}", exc_info=True)
            raise

    async def store_simulation_results(self, run_id: int, summary_df: pd.DataFrame, simulation_results: Dict, qualification_data: Dict):
        """
        Stores detailed simulation results, including team summaries and rank distributions.
        """
        logger.info(f"Storing simulation results for run_id: {run_id}")

        try:
            summary_insert_query = """
                INSERT INTO prediction_summary (
                    run_id, team_id, games_remaining, current_points, current_rank,
                    average_points, avg_final_rank, median_final_rank, best_rank, worst_rank,
                    rank_25, rank_75, best_points, worst_points, playoff_prob_pct, 
                    clinched, eliminated, created_at
                ) VALUES (
                    :run_id, :team_id, :games_remaining, :current_points, :current_rank,
                    :average_points, :avg_final_rank, :median_final_rank, :best_rank, :worst_rank,
                    :rank_25, :rank_75, :best_points, :worst_points, :playoff_prob_pct,
                    :clinched, :eliminated, NOW()
                )
                ON CONFLICT (run_id, team_id) DO UPDATE SET
                    games_remaining = EXCLUDED.games_remaining,
                    current_points = EXCLUDED.current_points,
                    current_rank = EXCLUDED.current_rank,
                    average_points = EXCLUDED.average_points,
                    avg_final_rank = EXCLUDED.avg_final_rank,
                    median_final_rank = EXCLUDED.median_final_rank,
                    best_rank = EXCLUDED.best_rank,
                    worst_rank = EXCLUDED.worst_rank,
                    rank_25 = EXCLUDED.rank_25,
                    rank_75 = EXCLUDED.rank_75,
                    best_points = EXCLUDED.best_points,
                    worst_points = EXCLUDED.worst_points,
                    playoff_prob_pct = EXCLUDED.playoff_prob_pct,
                    clinched = EXCLUDED.clinched,
                    eliminated = EXCLUDED.eliminated,
                    created_at = NOW()
            """
        
            for _, row in summary_df.iterrows():
                team_id = row['_team_id']
                rank_dist = simulation_results.get(team_id, [])
                qual_info = qualification_data.get(team_id, {})
            
                # Calculate rank statistics from actual simulation results
                if rank_dist:
                    rank_array = np.array(rank_dist)
                    median_rank = np.median(rank_array)
                    best_rank = np.min(rank_array)
                    worst_rank = np.max(rank_array)
                    rank_25 = np.percentile(rank_array, 25)
                    rank_75 = np.percentile(rank_array, 75)
                else:
                    # Fallback to DataFrame values if no simulation data
                    median_rank = self._safe_numeric(row.get('Median Final Rank', row['Average Final Rank']))
                    best_rank = self._safe_numeric(row.get('Best Rank', row['Average Final Rank']))
                    worst_rank = self._safe_numeric(row.get('Worst Rank', row['Average Final Rank']))
                    rank_25 = self._safe_numeric(row.get('_rank_25', row['Average Final Rank']))
                    rank_75 = self._safe_numeric(row.get('_rank_75', row['Average Final Rank']))
            
                # Determine if clinched/eliminated based on playoff probability
                playoff_prob = self._safe_numeric(row['Playoff Qualification %'], 0)
                clinched = playoff_prob >= 99.9
                eliminated = playoff_prob <= 0.1
            
                summary_values = {
                    "run_id": run_id,
                    "team_id": team_id,
                    "games_remaining": qual_info.get('games_remaining', 0),
                    "current_points": self._safe_numeric(row['Current Points'], 0, as_int=True),
                    "current_rank": self._safe_numeric(row.get('Current Rank'), None, as_int=True),
                    "average_points": self._safe_numeric(row['Average Points'], 0),
                    "avg_final_rank": self._safe_numeric(row['Average Final Rank']),
                    "median_final_rank": self._safe_numeric(median_rank),
                    "best_rank": self._safe_numeric(best_rank, as_int=True),
                    "worst_rank": self._safe_numeric(worst_rank, as_int=True),
                    "rank_25": self._safe_numeric(rank_25, as_int=True),
                    "rank_75": self._safe_numeric(rank_75, as_int=True),
                    "best_points": self._safe_numeric(row.get('Best Points'), as_int=True),
                    "worst_points": self._safe_numeric(row.get('Worst Points'), as_int=True),
                    "playoff_prob_pct": playoff_prob,
                    "clinched": clinched,
                    "eliminated": eliminated
                }
            
                await self.db.execute(summary_insert_query, values=summary_values)
        
            # Store shootout analysis data
            await self._store_shootout_analysis(run_id, qualification_data)
            
            logger.info(f"Stored {len(summary_df)} team prediction summaries for run_id: {run_id}")
        
        except Exception as e:
            logger.error(f"Database error in store_simulation_results for run_id {run_id}: {e}", exc_info=True)
            raise

    async def _store_shootout_analysis(self, run_id: int, qualification_data: Dict):
        """
        Store shootout analysis data. This provides data for the "Impact of Additional Shootout Wins" chart.
        """
        try:
            shootout_entries_stored = 0
            
            for team_id, qual_data in qualification_data.items():
                if not qual_data.get('shootout_win_impact'):
                    continue
                    
                shootout_impact = qual_data.get('shootout_win_impact', {})
                current_odds = qual_data.get('playoff_probability', 0)
                
                # Calculate wins needed for 50% and 75% playoff odds
                wins_for_50 = None
                wins_for_75 = None
                
                for additional_wins, projected_odds in shootout_impact.items():
                    if wins_for_50 is None and projected_odds >= 50:
                        wins_for_50 = additional_wins
                    if wins_for_75 is None and projected_odds >= 75:
                        wins_for_75 = additional_wins
                
                # Get corresponding summary_id from prediction_summary
                summary_query = """
                    SELECT summary_id FROM prediction_summary 
                    WHERE run_id = :run_id AND team_id = :team_id
                """
                summary_result = await self.db.fetch_one(
                    summary_query, 
                    values={"run_id": run_id, "team_id": team_id}
                )
                
                if not summary_result:
                    logger.warning(f"No summary_id found for team {team_id} in run {run_id}")
                    continue
                    
                # Insert or update shootout analysis
                shootout_query = """
                    INSERT INTO shootout_analysis (
                        summary_id, run_id, team_id, games_remaining,
                        wins_for_50_odds, wins_for_75_odds, current_odds
                    ) VALUES (
                        :summary_id, :run_id, :team_id, :games_remaining,
                        :wins_for_50_odds, :wins_for_75_odds, :current_odds
                    )
                    ON CONFLICT (summary_id) DO UPDATE SET
                        run_id = EXCLUDED.run_id,
                        team_id = EXCLUDED.team_id,
                        games_remaining = EXCLUDED.games_remaining,
                        wins_for_50_odds = EXCLUDED.wins_for_50_odds,
                        wins_for_75_odds = EXCLUDED.wins_for_75_odds,
                        current_odds = EXCLUDED.current_odds
                """
                
                values = {
                    "summary_id": summary_result['summary_id'],
                    "run_id": run_id,
                    "team_id": team_id,
                    "games_remaining": qual_data.get('games_remaining', 0),
                    "wins_for_50_odds": wins_for_50,
                    "wins_for_75_odds": wins_for_75,
                    "current_odds": current_odds
                }
                
                await self.db.execute(shootout_query, values=values)
                shootout_entries_stored += 1
                
            if shootout_entries_stored > 0:
                logger.info(f"Stored shootout analysis for {shootout_entries_stored} teams (run_id: {run_id})")
            else:
                logger.debug(f"No shootout analysis data to store for run_id: {run_id}")
            
        except Exception as e:
            logger.warning(f"Could not store shootout analysis for run_id {run_id}: {e}")
            # Don't raise - this is optional functionality that shouldn't break the main storage


    async def get_latest_simulation_run(self, user_id: int, conference: str, season_year: int) -> Optional[Dict]:
        """
        Retrieves the latest simulation run details for a given conference and season.
        """
        logger.debug(f"Fetching latest prediction run for user: {user_id}, conference: {conference}, season: {season_year}")
    
        try:
            conf_map = {"eastern": 1, "western": 2}
            conference_id = conf_map.get(conference.lower())
            
            if conference_id is None:
                logger.warning(f"Invalid conference: {conference}")
                return None
            
            query = """
                SELECT pr.*, us.user_id 
                FROM prediction_runs pr
                JOIN user_simulations us ON pr.run_id = us.run_id
                WHERE us.user_id = :user_id 
                AND pr.conference_id = :conference_id 
                AND pr.season_year = :season_year
                ORDER BY pr.run_date DESC 
                LIMIT 1
            """
            
            values = {
                "user_id": user_id,
                "conference_id": conference_id, 
                "season_year": season_year
            }
            
            run = await self.db.fetch_one(query, values=values)
            if run:
                logger.info(f"Found latest prediction run: ID {run['run_id']} for user: {user_id}")
                return dict(run)
            else:
                logger.info(f"No prediction run found for user: {user_id}, conference: {conference}")
                return None
                
        except Exception as e:
            logger.error(f"Database error in get_latest_simulation_run: {e}", exc_info=True)
            raise


    async def get_simulation_results_for_run(self, run_id: int) -> List[Dict]:
        """
        Retrieves all stored simulation results for a given run ID.
        """
        logger.debug(f"Fetching simulation results for run_id: {run_id}")
    
        try:
            query = """
                SELECT ps.*, t.team_name
                FROM prediction_summary ps
                JOIN team t ON ps.team_id = t.team_id
                WHERE ps.run_id = :run_id
                ORDER BY ps.avg_final_rank ASC
            """
            
            results = await self.db.fetch_all(query, values={"run_id": run_id})
            
            parsed_results = []
            for row in results:
                res_dict = dict(row)
                parsed_results.append(res_dict)
            
            logger.info(f"Found {len(parsed_results)} results for run_id: {run_id}")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Database error in get_simulation_results_for_run: {e}", exc_info=True)
            raise

    async def get_current_matchday(self, season_year: int) -> Optional[int]:
        """
        Calculate the current matchday based on completed games.
        """
        query = """
            SELECT MAX(matchday) as current_matchday
            FROM games
            WHERE season_year = :season_year
            AND is_completed = true
            AND matchday > 0
        """
        
        result = await self.db.fetch_one(query, values={"season_year": season_year})
        return result['current_matchday'] if result and result['current_matchday'] else 0

    async def find_rescheduled_games(self, season_year: int) -> List[Dict]:
        """
        Finds postponed games and looks for their rescheduled counterparts.
        """
        logger.info(f"Searching for rescheduled games in season {season_year}...")

        postponed_query = """
            SELECT * FROM games
            WHERE status = 'postponed' AND season_year = :season_year
        """
        postponed_games = await self.db.fetch_all(postponed_query, values={"season_year": season_year})

        rescheduled_games = []

        for postponed_game in postponed_games:
            scheduled_query = """
                SELECT * FROM games
                WHERE home_team_id = :home_team_id
                AND away_team_id = :away_team_id
                AND date > :postponed_date
                AND status = 'scheduled'
                ORDER BY date ASC
                LIMIT 1
            """

            scheduled_game = await self.db.fetch_one(
                scheduled_query,
                values={
                    "home_team_id": postponed_game['home_team_id'],
                    "away_team_id": postponed_game['away_team_id'],
                    "postponed_date": postponed_game['date']
                }
            )

            if scheduled_game:
                rescheduled_games.append({
                    "postponed_game": dict(postponed_game),
                    "rescheduled_game": dict(scheduled_game)
                })

        logger.info(f"Found {len(rescheduled_games)} potentially rescheduled games.")
        return rescheduled_games

    async def check_for_rescheduled_games(self, season_year: int):
        """Checks for and logs any rescheduled games."""
        logger.info("Checking for rescheduled games...")
        rescheduled_games = await self.find_rescheduled_games(season_year)

        if rescheduled_games:
            logger.info(f"Found {len(rescheduled_games)} potentially rescheduled games:")
            for game_pair in rescheduled_games:
                postponed = game_pair['postponed_game']
                rescheduled = game_pair['rescheduled_game']
                logger.info(f"  - Postponed: {postponed['game_id']} on {postponed['date'].strftime('%Y-%m-%d')}")
                logger.info(f"    Rescheduled to: {rescheduled['game_id']} on {rescheduled['date'].strftime('%Y-%m-%d')}")
        else:
            logger.info("No rescheduled games found.")
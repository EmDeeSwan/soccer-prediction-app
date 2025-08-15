import os
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
from src.common.utils import logger
from src.common import database
from src.common.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    This can be used to initialize resources or connections.
    """
    app.state.db_connected = False
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        logger.info("Attempting to connect to the database...")
        try:
            await database.connect()
            logger.info("Database connected successfully.")
            app.state.db_connected = True
        except asyncpg.exceptions.PostgresConnectionError as e:
            logger.error(f"Database connection failed during startup: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during database connection: {e}")
    else:
        logger.error("DATABASE_URL environment variable not set. Application will start without database connection.")
    
    yield

    if app.state.db_connected:
        logger.info("Shutting down database connection...")
        await database.disconnect()
        logger.info("Database disconnected.")
    else:
        logger.info("Skipping database disconnection as it was not connected.")

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://pkbipcas.com",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
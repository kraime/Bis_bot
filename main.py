import asyncio
import os
import sys
from loguru import logger
from config import LOGGING_CONFIG

from src.database import db
from src.embeddings import embedding_service
from src.vector_db import vector_db
from src.bot import bot_instance

# Configure loguru
logger.remove()  # Remove default handler

# Add console handler with colors
logger.add(
    sys.stderr,
    format=LOGGING_CONFIG["format"],
    level=LOGGING_CONFIG["level"],
    colorize=True
)

# Add file handler for detailed logs
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    format=LOGGING_CONFIG["file_format"],
    level="DEBUG",
    rotation=LOGGING_CONFIG["rotation"],
    retention=LOGGING_CONFIG["retention"],
    compression=LOGGING_CONFIG["compression"]
)


async def main():
    """Main function to run the bot"""
    try:
        # Initialize database connection
        logger.info("Connecting to database...")
        await db.connect()

        # Initialize vector database
        logger.info("Connecting to vector database...")
        await vector_db.initialize()

        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_service.load_model()

        # Setup bot
        logger.info("Setting up bot...")
        bot, dp = await bot_instance.setup_bot()

        logger.info("Starting bot...")
        # Start polling
        await dp.start_polling(bot, drop_pending_updates=True)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Simple test script for the Telegram Business Matching Bot
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()


async def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")

    try:
        from src.database import db
        print("✅ Database module imported")
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        return False

    try:
        from src.embeddings import embedding_service
        print("✅ Embeddings module imported")
    except Exception as e:
        print(f"❌ Embeddings import failed: {e}")
        return False

    try:
        from src.text_processing import text_processor
        print("✅ Text processing module imported")
    except Exception as e:
        print(f"❌ Text processing import failed: {e}")
        return False

    try:
        from src.llm_service import llm_service
        print("✅ LLM service module imported")
    except Exception as e:
        print(f"❌ LLM service import failed: {e}")
        return False

    return True


async def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing database connection...")
    try:
        from src.database import db
        await db.connect()
        print("✅ Database connection successful")

        # Test simple query with SQLite
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            async with conn.execute("SELECT 1") as cursor:
                result = await cursor.fetchone()
                if result and result[0] == 1:
                    print("✅ Database query test passed")
                else:
                    print("❌ Database query test failed")

        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def test_embedding_model():
    """Test embedding model loading"""
    print("\n🔍 Testing embedding model...")
    try:
        from src.embeddings import embedding_service
        embedding_service.load_model()
        print("✅ Embedding model loaded successfully")

        # Test embedding creation
        test_text = "Я работаю в сфере IT"
        embedding = embedding_service.create_text_embedding(test_text)
        print(f"✅ Test embedding created, dimension: {len(embedding)}")

        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False


async def test_text_processing():
    """Test text processing"""
    print("\n🔍 Testing text processing...")
    try:
        from src.text_processing import text_processor

        # Test text cleaning
        dirty_text = "  Привет,   мир!  "
        clean_text = text_processor.clean_text(dirty_text)
        print(f"✅ Text cleaning: '{dirty_text}' -> '{clean_text}'")

        # Test keyword extraction
        text = "Я работаю в сфере информационных технологий и разработки программного обеспечения"
        keywords = text_processor.extract_keywords(text)
        print(f"✅ Keywords extracted: {keywords}")

        # Test profile preparation
        answer_1 = "Я разработчик"
        answer_2 = "Ищу партнеров"
        answer_3 = "Могу помочь с кодом"

        processed = text_processor.prepare_profile_text(answer_1, answer_2, answer_3)
        print(f"✅ Profile processed, keywords: {processed['keywords']}")

        return True
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Starting Simple Bot Tests\n")

    # Test imports
    if not await test_imports():
        print("\n❌ Import tests failed. Please check your Python path and module structure.")
        return

    # Test database
    if not await test_database_connection():
        print("\n❌ Database tests failed. Please check your DATABASE_URL in .env file.")
        return

    # Test embeddings
    if not await test_embedding_model():
        print("\n❌ Embedding tests failed. This might be due to missing dependencies.")
        return

    # Test text processing
    if not await test_text_processing():
        print("\n❌ Text processing tests failed.")
        return

    print("\n🎉 All basic tests passed!")
    print("\nYou can now test the bot manually in Telegram:")
    print("1. Find @BM_Skolkovo_bot")
    print("2. Send /start")
    print("3. Complete the profile creation")
    print("4. Try finding matches")

    # Cleanup
    try:
        from src.database import db
        await db.close()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())
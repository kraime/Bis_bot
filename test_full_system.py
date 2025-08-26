#!/usr/bin/env python3
"""
Full system test for SQLite + Qdrant setup
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()


async def test_full_system():
    """Test the complete system"""
    print("🚀 Testing Full System (SQLite + Qdrant)\n")

    try:
        # Import modules
        from src.database import db
        from src.embeddings import embedding_service
        from src.vector_db import vector_db
        from src.text_processing import text_processor

        # Initialize database
        print("🔍 Initializing SQLite database...")
        await db.connect()
        print("✅ SQLite initialized")

        # Initialize Qdrant
        print("🔍 Initializing Qdrant...")
        try:
            await vector_db.initialize()
            print("✅ Qdrant initialized")
        except Exception as e:
            print(f"❌ Qdrant failed: {e}")
            print("Please start Qdrant first:")
            print("1. Download from: https://github.com/qdrant/qdrant/releases")
            print("2. Run: qdrant")
            return False

        # Load embedding model
        print("🔍 Loading embedding model...")
        embedding_service.load_model()
        print("✅ Embedding model loaded")

        # Test user creation
        print("🔍 Testing user creation...")
        test_user = await db.get_or_create_user(
            telegram_id=12345,
            username="test_user",
            first_name="Test",
            last_name="User"
        )
        print(f"✅ User created: {test_user['first_name']}")

        # Test profile creation
        print("🔍 Testing profile creation...")
        answer_1 = "Я работаю в сфере IT, занимаюсь разработкой веб-приложений"
        answer_2 = "Ищу партнеров для совместных проектов"
        answer_3 = "Могу помочь с техническими вопросами"

        # Process text
        processed_data = text_processor.prepare_profile_text(answer_1, answer_2, answer_3)

        # Create embedding
        embedding = embedding_service.create_profile_embedding(
            processed_data['clean_answers']['answer_1'],
            processed_data['clean_answers']['answer_2'],
            processed_data['clean_answers']['answer_3']
        )

        # Save to SQLite
        await db.save_user_profile(
            test_user['id'],
            processed_data['clean_answers']['answer_1'],
            processed_data['clean_answers']['answer_2'],
            processed_data['clean_answers']['answer_3'],
            embedding,
            processed_data['keywords']
        )
        print("✅ Profile saved to SQLite")

        # Save to Qdrant
        profile_data = {
            "telegram_id": test_user['telegram_id'],
            "username": test_user['username'],
            "first_name": test_user['first_name'],
            "last_name": test_user['last_name'],
            "answer_1": processed_data['clean_answers']['answer_1'],
            "answer_2": processed_data['clean_answers']['answer_2'],
            "answer_3": processed_data['clean_answers']['answer_3'],
            "keywords": processed_data['keywords']
        }

        vector_db.save_profile_embedding(test_user['id'], embedding, profile_data)
        print("✅ Profile saved to Qdrant")

        # Test vector search
        print("🔍 Testing vector search...")
        search_embedding = embedding_service.create_text_embedding("IT разработчик ищет партнеров")
        similar_profiles = vector_db.search_similar_profiles(search_embedding, test_user['id'], limit=5)
        print(f"✅ Vector search completed, found {len(similar_profiles)} profiles")

        # Get collection info
        info = vector_db.get_collection_info()
        print(f"📊 Qdrant collection info: {info}")

        print("\n🎉 Full system test passed!")
        print("\nSystem is ready! You can now:")
        print("1. Run: python main.py")
        print("2. Test the bot in Telegram")

        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False
    finally:
        try:
            await db.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_full_system())
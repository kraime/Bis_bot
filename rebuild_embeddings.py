#!/usr/bin/env python3
"""
Rebuild embeddings for existing users
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()


async def rebuild_embeddings():
    """Rebuild embeddings for all users who don't have them in Qdrant"""
    print("🔄 Rebuilding embeddings for existing users...\n")

    try:
        from src.database import db
        from src.embeddings import embedding_service
        from src.vector_db import vector_db
        from src.text_processing import text_processor

        # Initialize systems
        await db.connect()
        await vector_db.initialize()
        embedding_service.load_model()

        # Get all users with profiles
        all_profiles = await db.get_all_active_profiles()
        print(f"📊 Found {len(all_profiles)} active profiles")

        rebuilt_count = 0

        for profile in all_profiles:
            user_id = profile['user_id']
            telegram_id = profile['telegram_id']
            first_name = profile.get('first_name', 'Unknown')

            try:
                # Check if user exists in Qdrant
                existing_point = vector_db.client.retrieve(
                    collection_name=vector_db.collection_name,
                    ids=[user_id]
                )

                if existing_point:
                    print(f"✅ User {user_id} ({first_name}): already has embedding")
                    continue

                print(f"🔄 User {user_id} ({first_name}): creating embedding...")

                # Process text
                processed_data = text_processor.prepare_profile_text(
                    profile['answer_1'],
                    profile['answer_2'],
                    profile['answer_3']
                )

                # Create embedding
                embedding = embedding_service.create_profile_embedding(
                    processed_data['clean_answers']['answer_1'],
                    processed_data['clean_answers']['answer_2'],
                    processed_data['clean_answers']['answer_3']
                )

                # Save to Qdrant
                profile_data = {
                    "telegram_id": profile['telegram_id'],
                    "username": profile['username'],
                    "first_name": profile['first_name'],
                    "last_name": profile['last_name'],
                    "answer_1": profile['answer_1'],
                    "answer_2": profile['answer_2'],
                    "answer_3": profile['answer_3'],
                    "keywords": profile.get('keywords', [])
                }

                vector_db.save_profile_embedding(user_id, embedding, profile_data)
                rebuilt_count += 1

                print(f"✅ User {user_id} ({first_name}): embedding created and saved")

            except Exception as e:
                print(f"❌ User {user_id} ({first_name}): failed to create embedding - {e}")

        print(f"\n🎉 Rebuild complete!")
        print(f"📊 Processed: {len(all_profiles)} profiles")
        print(f"🔄 Rebuilt: {rebuilt_count} embeddings")

        # Show collection info
        info = vector_db.get_collection_info()
        print(f"📈 Qdrant collection: {info.get('points_count', 0)} total profiles")

    except Exception as e:
        print(f"❌ Rebuild failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await db.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(rebuild_embeddings())
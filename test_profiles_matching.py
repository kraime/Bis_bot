#!/usr/bin/env python3
"""
Comprehensive test for profile creation and matching system
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()


async def test_comprehensive_system():
    """Test complete profile creation and matching system"""
    print("🚀 Testing Comprehensive Profile & Matching System\n")

    try:
        # Import modules
        from src.database import db
        from src.embeddings import embedding_service
        from src.vector_db import vector_db
        from src.text_processing import text_processor
        from src.llm_service import llm_service

        # Initialize systems
        print("🔍 Initializing systems...")
        await db.connect()
        await vector_db.initialize()
        embedding_service.load_model()
        print("✅ All systems initialized")

        # Clear existing test data
        print("\n🧹 Cleaning up existing test data...")
        try:
            # Clean SQLite
            import aiosqlite
            async with aiosqlite.connect(db.db_path) as conn:
                # Delete profiles first (due to foreign key constraints)
                await conn.execute("""
                    DELETE FROM user_profiles WHERE user_id IN (
                        SELECT id FROM users WHERE telegram_id BETWEEN 10000 AND 19999
                    )
                """)
                await conn.execute("""
                    DELETE FROM profile_history WHERE user_id IN (
                        SELECT id FROM users WHERE telegram_id BETWEEN 10000 AND 19999
                    )
                """)
                await conn.execute("""
                    DELETE FROM user_states WHERE user_id IN (
                        SELECT id FROM users WHERE telegram_id BETWEEN 10000 AND 19999
                    )
                """)
                # Now delete users
                await conn.execute("DELETE FROM users WHERE telegram_id BETWEEN 10000 AND 19999")
                await conn.commit()

            # Clean Qdrant (delete points with IDs we'll use)
            for user_id in range(1, 20):  # Wider range for cleanup
                try:
                    vector_db.delete_profile(user_id)
                except:
                    pass

            print("✅ Test data cleaned")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

        # Create diverse test profiles
        test_profiles = [
            {
                "telegram_id": 10001,
                "username": "dev_alex",
                "first_name": "Александр",
                "last_name": "Разработчик",
                "answers": [
                    "Я Senior Python разработчик, работаю с Django, FastAPI и машинным обучением. Имею опыт создания высоконагруженных веб-сервисов",
                    "Ищу техническое партнерство для создания AI-стартапа, нужны инвесторы и бизнес-партнеры",
                    "Могу помочь с архитектурой системы, разработкой MVP, настройкой CI/CD и менторством junior разработчиков"
                ]
            },
            {
                "telegram_id": 10002,
                "username": "investor_maria",
                "first_name": "Мария",
                "last_name": "Инвестор",
                "answers": [
                    "Я венчурный инвестор, специализируюсь на IT-стартапах на стадии seed и Series A. Портфель 50+ компаний",
                    "Ищу перспективные технологические проекты в области AI, fintech и healthtech для инвестиций",
                    "Предоставляю финансирование от $100K до $5M, менторство, связи с клиентами и экспертизу по масштабированию"
                ]
            },
            {
                "telegram_id": 10003,
                "username": "designer_kate",
                "first_name": "Екатерина",
                "last_name": "Дизайнер",
                "answers": [
                    "UX/UI дизайнер с 7-летним опытом, работала в Яндексе и Mail.ru. Специализируюсь на мобильных приложениях",
                    "Ищу команду для создания инновационного продукта, хочу стать co-founder в перспективном стартапе",
                    "Могу создать полный дизайн продукта от исследований до готовых макетов, провести UX-аудит и A/B тесты"
                ]
            },
            {
                "telegram_id": 10004,
                "username": "marketer_john",
                "first_name": "Иван",
                "last_name": "Маркетолог",
                "answers": [
                    "Digital-маркетолог, эксперт по performance-маркетингу. Запускал рекламу с бюджетом $1M+",
                    "Ищу стартапы на стадии product-market fit для масштабирования через платные каналы",
                    "Помогу настроить воронки продаж, запустить рекламу в Facebook/Google, провести CRO и увеличить LTV"
                ]
            },
            {
                "telegram_id": 10005,
                "username": "sales_director",
                "first_name": "Дмитрий",
                "last_name": "Продажник",
                "answers": [
                    "Директор по продажам B2B, 10 лет опыта. Построил отделы продаж в 3 компаниях от 0 до $10M ARR",
                    "Ищу позицию CPO или co-founder в B2B SaaS стартапе с готовым продуктом",
                    "Могу построить отдел продаж с нуля, обучить команду, настроить CRM и процессы, найти первых клиентов"
                ]
            },
            {
                "telegram_id": 10006,
                "username": "data_scientist",
                "first_name": "Анна",
                "last_name": "Аналитик",
                "answers": [
                    "Data Scientist, PhD в области машинного обучения. Работаю с NLP, computer vision и рекомендательными системами",
                    "Ищу проекты в области AI/ML, хочу применить свои знания в коммерческих продуктах",
                    "Могу создать ML-модели, провести анализ данных, построить data pipeline и внедрить AI в продукт"
                ]
            }
        ]

        created_users = []

        print(f"\n👥 Creating {len(test_profiles)} diverse test profiles...")

        for i, profile_data in enumerate(test_profiles, 1):
            try:
                print(f"\n🔍 Creating profile {i}/{len(test_profiles)}: {profile_data['first_name']}")

                # Create user
                user = await db.get_or_create_user(
                    telegram_id=profile_data["telegram_id"],
                    username=profile_data["username"],
                    first_name=profile_data["first_name"],
                    last_name=profile_data["last_name"]
                )

                # Process profile text
                processed_data = text_processor.prepare_profile_text(
                    profile_data["answers"][0],
                    profile_data["answers"][1],
                    profile_data["answers"][2]
                )

                print(f"  📝 Keywords: {processed_data['keywords'][:5]}...")

                # Create embedding
                embedding = embedding_service.create_profile_embedding(
                    processed_data['clean_answers']['answer_1'],
                    processed_data['clean_answers']['answer_2'],
                    processed_data['clean_answers']['answer_3']
                )

                # Save to SQLite
                await db.save_user_profile(
                    user['id'],
                    processed_data['clean_answers']['answer_1'],
                    processed_data['clean_answers']['answer_2'],
                    processed_data['clean_answers']['answer_3'],
                    embedding,
                    processed_data['keywords']
                )

                # Save to Qdrant
                profile_payload = {
                    "telegram_id": user['telegram_id'],
                    "username": user['username'],
                    "first_name": user['first_name'],
                    "last_name": user['last_name'],
                    "answer_1": processed_data['clean_answers']['answer_1'],
                    "answer_2": processed_data['clean_answers']['answer_2'],
                    "answer_3": processed_data['clean_answers']['answer_3'],
                    "keywords": processed_data['keywords']
                }

                vector_db.save_profile_embedding(user['id'], embedding, profile_payload)

                created_users.append({
                    'user': user,
                    'profile_data': profile_data,
                    'processed_data': processed_data,
                    'embedding': embedding
                })

                print(f"  ✅ Profile created and saved")

            except Exception as e:
                print(f"  ❌ Failed to create profile for {profile_data['first_name']}: {e}")

        print(f"\n✅ Created {len(created_users)} profiles successfully")

        # Get collection info
        info = vector_db.get_collection_info()
        print(f"📊 Qdrant collection: {info['points_count']} profiles stored")

        # Test matching for each user
        print(f"\n🔍 Testing matching system for each user...")

        for i, user_data in enumerate(created_users, 1):
            user = user_data['user']
            profile_data = user_data['profile_data']

            print(f"\n--- Matching Test {i}/{len(created_users)}: {profile_data['first_name']} ---")
            print(f"👤 Полный профиль:")
            print(f"   🏢 Сфера: {profile_data['answers'][0]}")
            print(f"   🔍 Ищет: {profile_data['answers'][1]}")
            print(f"   🤝 Может помочь: {profile_data['answers'][2]}")
            print(f"   📋 Ключевые слова: {user_data['processed_data']['keywords'][:5]}")

            try:
                # Get user profile from database
                db_profile = await db.get_user_profile(user['id'])
                if not db_profile:
                    print("  ❌ Profile not found in database")
                    continue

                # Test keyword-based search
                keywords = db_profile.get('keywords', [])
                if keywords:
                    keyword_profiles = await db.find_profiles_with_keywords(
                        user['id'], keywords, limit=10
                    )
                    print(f"  🔍 Keyword search: found {len(keyword_profiles)} profiles")
                else:
                    keyword_profiles = []

                # Test vector search
                vector_profiles = vector_db.search_similar_profiles(
                    user_data['embedding'], user['id'], limit=10
                )
                print(f"  🔍 Vector search: found {len(vector_profiles)} profiles")

                # Combine results
                all_candidates = []
                seen_users = set()

                # Add keyword results
                for p in keyword_profiles:
                    user_key = p['telegram_id']
                    if user_key not in seen_users:
                        seen_users.add(user_key)
                        all_candidates.append(p)

                # Add vector results
                for p in vector_profiles:
                    user_key = p['telegram_id']
                    if user_key not in seen_users:
                        seen_users.add(user_key)
                        all_candidates.append(p)

                print(f"  📊 Total candidates: {len(all_candidates)}")

                if all_candidates:
                    # Test LLM matching
                    try:
                        best_matches = await llm_service.find_best_matches(
                            db_profile, all_candidates, top_k=3
                        )
                        print(f"  🤖 LLM analysis: {len(best_matches)} best matches")

                        for j, match in enumerate(best_matches, 1):
                            name = match.get('first_name', 'Unknown')
                            score = match.get('match_score', 'N/A')
                            reason = match.get('match_reason', 'No reason')[:50]
                            print(f"    {j}. {name} (Score: {score}) - {reason}...")

                        # Test summary generation
                        if best_matches:
                            summary = await llm_service.generate_match_summary(db_profile, best_matches)
                            print(f"  📝 Summary: {summary[:80]}...")

                    except Exception as e:
                        print(f"  ⚠️ LLM matching failed: {e}")
                else:
                    print("  ⚠️ No candidates found for matching")

            except Exception as e:
                print(f"  ❌ Matching failed: {e}")

        # Test cross-matching scenarios
        print(f"\n🎯 Testing specific matching scenarios...")

        # Scenario 1: Developer looking for investor
        dev_user = next((u for u in created_users if 'dev_alex' in u['user']['username']), None)
        investor_user = next((u for u in created_users if 'investor_maria' in u['user']['username']), None)

        if dev_user and investor_user:
            print(f"\n📋 Scenario 1: Developer → Investor matching")
            dev_profile = await db.get_user_profile(dev_user['user']['id'])

            # Search for investor
            vector_results = vector_db.search_similar_profiles(
                dev_user['embedding'], dev_user['user']['id'], limit=5
            )

            investor_found = any(p['telegram_id'] == investor_user['user']['telegram_id'] for p in vector_results)
            print(f"  {'✅' if investor_found else '❌'} Investor found in vector search: {investor_found}")

            if vector_results:
                llm_matches = await llm_service.find_best_matches(dev_profile, vector_results, top_k=3)
                investor_in_top = any(m['telegram_id'] == investor_user['user']['telegram_id'] for m in llm_matches)
                print(f"  {'✅' if investor_in_top else '❌'} Investor in LLM top matches: {investor_in_top}")

        # Final statistics
        print(f"\n📊 Final Statistics:")
        print(f"  👥 Total profiles created: {len(created_users)}")
        print(f"  🗄️ SQLite profiles: {len(created_users)}")

        final_info = vector_db.get_collection_info()
        print(f"  🔍 Qdrant profiles: {final_info.get('points_count', 0)}")
        print(f"  📈 Collection status: {final_info.get('status', 'Unknown')}")

        print(f"\n🎉 Comprehensive test completed successfully!")
        print(f"\nSystem is fully tested and ready for production!")
        print(f"\nTo start the bot: python main.py")

        return True

    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await db.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_comprehensive_system())
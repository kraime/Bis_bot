#!/usr/bin/env python3
"""
Script to delete users from both SQLite and Qdrant databases
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()


async def delete_user_by_telegram_id(telegram_id: int, confirm: bool = False):
    """Delete user by Telegram ID from both databases"""
    print(f"🔍 Looking for user with Telegram ID: {telegram_id}")

    try:
        from src.database import db
        from src.vector_db import vector_db

        # Initialize databases
        await db.connect()
        await vector_db.initialize()

        # Find user in SQLite
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            async with conn.execute(
                    "SELECT id, telegram_id, username, first_name, last_name FROM users WHERE telegram_id = ?",
                    (telegram_id,)
            ) as cursor:
                user = await cursor.fetchone()

        if not user:
            print(f"❌ User with Telegram ID {telegram_id} not found in database")
            return False

        user_id, tg_id, username, first_name, last_name = user
        user_display = f"{first_name or 'Unknown'}"
        if username:
            user_display += f" (@{username})"
        user_display += f" (ID: {tg_id})"

        print(f"👤 Found user: {user_display}")
        print(f"   Database ID: {user_id}")

        # Get user profile info
        async with aiosqlite.connect(db.db_path) as conn:
            async with conn.execute(
                    "SELECT answer_1, answer_2, answer_3 FROM user_profiles WHERE user_id = ?",
                    (user_id,)
            ) as cursor:
                profile = await cursor.fetchone()

        if profile:
            print(f"   Profile: {profile[0][:50]}...")
        else:
            print("   No profile found")

        # Check if user exists in Qdrant
        try:
            qdrant_point = vector_db.client.retrieve(
                collection_name=vector_db.collection_name,
                ids=[user_id]
            )
            has_qdrant_profile = len(qdrant_point) > 0
            print(f"   Qdrant profile: {'Yes' if has_qdrant_profile else 'No'}")
        except Exception as e:
            print(f"   Qdrant check failed: {e}")
            has_qdrant_profile = False

        if not confirm:
            print(f"\n⚠️  This will permanently delete:")
            print(f"   • User record from SQLite")
            print(f"   • User profile from SQLite")
            print(f"   • Profile history from SQLite")
            print(f"   • User states from SQLite")
            if has_qdrant_profile:
                print(f"   • Vector embedding from Qdrant")

            confirmation = input(f"\nAre you sure you want to delete {user_display}? (yes/no): ")
            if confirmation.lower() not in ['yes', 'y']:
                print("❌ Deletion cancelled")
                return False

        # Delete from SQLite (cascade will handle related records)
        async with aiosqlite.connect(db.db_path) as conn:
            await conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            await conn.commit()

        print(f"✅ Deleted user from SQLite: {user_display}")

        # Delete from Qdrant
        if has_qdrant_profile:
            try:
                vector_db.delete_profile(user_id)
                print(f"✅ Deleted user from Qdrant: {user_display}")
            except Exception as e:
                print(f"⚠️ Failed to delete from Qdrant: {e}")

        print(f"🎉 User {user_display} deleted successfully!")
        return True

    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return False
    finally:
        try:
            await db.close()
        except:
            pass


async def delete_user_by_name(name: str, confirm: bool = False):
    """Delete user by name (first_name or username)"""
    print(f"🔍 Looking for users with name containing: '{name}'")

    try:
        from src.database import db
        from src.vector_db import vector_db

        # Initialize databases
        await db.connect()
        await vector_db.initialize()

        # Find users in SQLite
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            async with conn.execute(
                    """SELECT id, telegram_id, username, first_name, last_name 
                       FROM users 
                       WHERE first_name LIKE ? OR username LIKE ? OR last_name LIKE ?""",
                    (f"%{name}%", f"%{name}%", f"%{name}%")
            ) as cursor:
                users = await cursor.fetchall()

        if not users:
            print(f"❌ No users found with name containing '{name}'")
            return False

        print(f"👥 Found {len(users)} user(s):")
        for i, user in enumerate(users, 1):
            user_id, tg_id, username, first_name, last_name = user
            user_display = f"{first_name or 'Unknown'}"
            if username:
                user_display += f" (@{username})"
            user_display += f" (ID: {tg_id})"
            print(f"   {i}. {user_display}")

        if len(users) == 1:
            # Single user found, delete directly
            user = users[0]
            return await delete_user_by_telegram_id(user[1], confirm)
        else:
            # Multiple users found, ask which one
            try:
                choice = int(input(f"\nWhich user to delete? (1-{len(users)}, 0 to cancel): "))
                if choice == 0:
                    print("❌ Deletion cancelled")
                    return False
                elif 1 <= choice <= len(users):
                    user = users[choice - 1]
                    return await delete_user_by_telegram_id(user[1], confirm)
                else:
                    print("❌ Invalid choice")
                    return False
            except ValueError:
                print("❌ Invalid input")
                return False

    except Exception as e:
        print(f"❌ Error searching users: {e}")
        return False
    finally:
        try:
            await db.close()
        except:
            pass


async def list_all_users():
    """List all users in the database"""
    print("👥 All users in database:")

    try:
        from src.database import db

        # Initialize database
        await db.connect()

        # Get all users
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            async with conn.execute(
                    """SELECT u.id, u.telegram_id, u.username, u.first_name, u.last_name, u.created_at,
                              CASE WHEN up.user_id IS NOT NULL THEN 'Yes' ELSE 'No' END as has_profile
                       FROM users u
                       LEFT JOIN user_profiles up ON u.id = up.user_id
                       ORDER BY u.created_at DESC"""
            ) as cursor:
                users = await cursor.fetchall()

        if not users:
            print("📭 No users found in database")
            return

        print(f"📊 Total users: {len(users)}\n")

        for user in users:
            user_id, tg_id, username, first_name, last_name, created_at, has_profile = user
            user_display = f"{first_name or 'Unknown'}"
            if username:
                user_display += f" (@{username})"

            print(f"👤 {user_display}")
            print(f"   Telegram ID: {tg_id}")
            print(f"   Database ID: {user_id}")
            print(f"   Profile: {has_profile}")
            print(f"   Created: {created_at}")
            print()

    except Exception as e:
        print(f"❌ Error listing users: {e}")
    finally:
        try:
            await db.close()
        except:
            pass


async def cleanup_orphaned_data():
    """Clean up orphaned data in both databases"""
    print("🧹 Cleaning up orphaned data...")

    try:
        from src.database import db
        from src.vector_db import vector_db

        # Initialize databases
        await db.connect()
        await vector_db.initialize()

        # Get all user IDs from SQLite
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            async with conn.execute("SELECT id FROM users") as cursor:
                sqlite_user_ids = {row[0] for row in await cursor.fetchall()}

        print(f"📊 SQLite users: {len(sqlite_user_ids)}")

        # Get all points from Qdrant
        try:
            qdrant_info = vector_db.get_collection_info()
            print(f"📊 Qdrant profiles: {qdrant_info.get('points_count', 0)}")

            # Get all point IDs from Qdrant
            scroll_result = vector_db.client.scroll(
                collection_name=vector_db.collection_name,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )

            qdrant_user_ids = {point.id for point in scroll_result[0]}
            print(f"📊 Qdrant point IDs: {len(qdrant_user_ids)}")

            # Find orphaned Qdrant profiles (exist in Qdrant but not in SQLite)
            orphaned_qdrant = qdrant_user_ids - sqlite_user_ids

            if orphaned_qdrant:
                print(f"🗑️ Found {len(orphaned_qdrant)} orphaned Qdrant profiles: {list(orphaned_qdrant)}")

                confirm = input("Delete orphaned Qdrant profiles? (yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    for user_id in orphaned_qdrant:
                        try:
                            vector_db.delete_profile(user_id)
                            print(f"✅ Deleted orphaned Qdrant profile: {user_id}")
                        except Exception as e:
                            print(f"❌ Failed to delete Qdrant profile {user_id}: {e}")
                else:
                    print("❌ Cleanup cancelled")
            else:
                print("✅ No orphaned Qdrant profiles found")

        except Exception as e:
            print(f"⚠️ Qdrant cleanup failed: {e}")

        print("🎉 Cleanup completed!")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
    finally:
        try:
            await db.close()
        except:
            pass


async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🗑️ User Deletion Script")
        print("\nUsage:")
        print("  python delete_user.py <telegram_id>           # Delete by Telegram ID")
        print("  python delete_user.py --name <name>           # Delete by name")
        print("  python delete_user.py --list                  # List all users")
        print("  python delete_user.py --cleanup               # Clean orphaned data")
        print("  python delete_user.py --force <telegram_id>   # Delete without confirmation")
        print("\nExamples:")
        print("  python delete_user.py 123456789")
        print("  python delete_user.py --name Alexis")
        print("  python delete_user.py --force 123456789")
        return

    command = sys.argv[1]

    if command == "--list":
        await list_all_users()
    elif command == "--cleanup":
        await cleanup_orphaned_data()
    elif command == "--name":
        if len(sys.argv) < 3:
            print("❌ Please provide a name to search for")
            return
        name = sys.argv[2]
        await delete_user_by_name(name)
    elif command == "--force":
        if len(sys.argv) < 3:
            print("❌ Please provide a Telegram ID")
            return
        try:
            telegram_id = int(sys.argv[2])
            await delete_user_by_telegram_id(telegram_id, confirm=True)
        except ValueError:
            print("❌ Invalid Telegram ID")
    else:
        # Assume it's a Telegram ID
        try:
            telegram_id = int(command)
            await delete_user_by_telegram_id(telegram_id)
        except ValueError:
            print("❌ Invalid Telegram ID")


if __name__ == "__main__":
    asyncio.run(main())
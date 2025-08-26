#!/usr/bin/env python3
"""
Migration script to add birthday column to existing database
"""

import asyncio
import aiosqlite
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def migrate_database():
    """Add birthday column to users table"""
    database_path = os.getenv("DATABASE_PATH", "business_bot.db")

    print(f"🔄 Migrating database: {database_path}")

    try:
        async with aiosqlite.connect(database_path) as db:
            # Check if birthday column already exists
            async with db.execute("PRAGMA table_info(users)") as cursor:
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]

                if 'birthday' in column_names:
                    print("✅ Birthday column already exists")
                    return True

            # Add birthday column
            print("📅 Adding birthday column...")
            await db.execute("ALTER TABLE users ADD COLUMN birthday TEXT")
            await db.commit()

            print("✅ Birthday column added successfully")

            # Verify the column was added
            async with db.execute("PRAGMA table_info(users)") as cursor:
                columns = await cursor.fetchall()
                print("📊 Current table structure:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")

            return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


async def main():
    """Main migration function"""
    print("🚀 Starting database migration...\n")

    success = await migrate_database()

    if success:
        print("\n🎉 Migration completed successfully!")
        print("You can now run the chunking test.")
    else:
        print("\n❌ Migration failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
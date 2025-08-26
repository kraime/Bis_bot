import aiosqlite
from typing import Optional, List, Dict, Any
import json
from datetime import datetime, timedelta
from loguru import logger
from config import DATABASE_PATH


class Database:
    def __init__(self):
        self.db_path = DATABASE_PATH

    async def connect(self):
        """Initialize database and create tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await db.commit()
            logger.info(f"SQLite database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close database connection (SQLite doesn't need explicit closing)"""
        logger.info("Database connection closed")

    async def _create_tables(self, db):
        """Create database tables"""
        # Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER UNIQUE NOT NULL,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                phone TEXT,
                birthday TEXT,
                birthday TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_profile_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        # User profiles table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                answer_1 TEXT NOT NULL,
                answer_2 TEXT NOT NULL,
                answer_3 TEXT NOT NULL,
                keywords TEXT, -- JSON array as text
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id)
            )
        """)

        # Profile history table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS profile_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                answer_1 TEXT,
                answer_2 TEXT,
                answer_3 TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User states table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                state TEXT NOT NULL,
                data TEXT, -- JSON as text
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id)
            )
        """)

        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON user_profiles(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON profile_history(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_states_user_id ON user_states(user_id)")

    async def get_or_create_user(self, telegram_id: int, username: str = None,
                                 first_name: str = None, last_name: str = None) -> Dict[str, Any]:
        """Get existing user or create new one"""
        async with aiosqlite.connect(self.db_path) as db:
            # Try to get existing user
            async with db.execute(
                    "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
            ) as cursor:
                user = await cursor.fetchone()

            if user:
                # Update user info if changed
                await db.execute(
                    """UPDATE users SET username = ?, first_name = ?, last_name = ?, 
                       updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?""",
                    (username, first_name, last_name, telegram_id)
                )
                await db.commit()

                # Convert to dict
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return dict(zip(columns, user)) if columns else {
                    'id': user[0], 'telegram_id': user[1], 'username': user[2],
                    'first_name': user[3], 'last_name': user[4]
                }
            else:
                # Create new user with race condition handling
                try:
                    async with db.execute(
                            """INSERT INTO users (telegram_id, username, first_name, last_name) 
                               VALUES (?, ?, ?, ?)""",
                            (telegram_id, username, first_name, last_name)
                    ) as cursor:
                        user_id = cursor.lastrowid
                    await db.commit()

                    return {
                        'id': user_id,
                        'telegram_id': telegram_id,
                        'username': username,
                        'first_name': first_name,
                        'last_name': last_name
                    }
                except aiosqlite.IntegrityError as e:
                    if "UNIQUE constraint failed: users.telegram_id" in str(e):
                        # User was created by another concurrent request, fetch it
                        logger.warning(f"Race condition detected for telegram_id {telegram_id}, fetching existing user")
                        async with db.execute(
                                "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
                        ) as cursor:
                            user = await cursor.fetchone()
                            if user:
                                columns = ['id', 'telegram_id', 'username', 'first_name', 'last_name', 'phone',
                                           'created_at', 'updated_at', 'last_profile_update', 'is_active']
                                return dict(zip(columns, user))
                    raise  # Re-raise if it's a different integrity error

    async def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user profile by user_id"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    profile = dict(zip(columns, row))
                    # Parse keywords JSON
                    if profile.get('keywords'):
                        try:
                            profile['keywords'] = json.loads(profile['keywords'])
                        except:
                            profile['keywords'] = []
                    return profile
                return None

    async def save_user_profile(self, user_id: int, answer_1: str, answer_2: str,
                                answer_3: str, embedding: List[float] = None, keywords: List[str] = None):
        """Save or update user profile"""
        logger.info(f"💾 SAVING PROFILE | User ID {user_id}: starting save process")
        async with aiosqlite.connect(self.db_path) as db:
            # Check if profile exists
            async with db.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ) as cursor:
                existing = await cursor.fetchone()

            keywords_json = json.dumps(keywords or [])

            if existing:
                logger.info(f"💾 UPDATING PROFILE | User ID {user_id}: updating existing profile")
                # Save current profile to history
                await db.execute(
                    """INSERT INTO profile_history (user_id, answer_1, answer_2, answer_3, keywords)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, existing[2], existing[3], existing[4], existing[5])
                )

                # Update current profile
                await db.execute(
                    """UPDATE user_profiles SET answer_1 = ?, answer_2 = ?, answer_3 = ?,
                       keywords = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?""",
                    (answer_1, answer_2, answer_3, keywords_json, user_id)
                )
            else:
                logger.info(f"💾 CREATING PROFILE | User ID {user_id}: creating new profile")
                # Create new profile
                await db.execute(
                    """INSERT INTO user_profiles (user_id, answer_1, answer_2, answer_3, keywords)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, answer_1, answer_2, answer_3, keywords_json)
                )

            # Update user's last profile update time
            await db.execute(
                "UPDATE users SET last_profile_update = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            await db.commit()

            logger.info(f"💾 PROFILE SAVED | User ID {user_id}: {'updated' if existing else 'created'} successfully")
            return existing is not None  # Return True if updated, False if created

    async def get_user_state(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user's current conversation state"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                    "SELECT * FROM user_states WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    state = dict(zip(columns, row))
                    # Parse data JSON
                    if state.get('data'):
                        try:
                            state['data'] = json.loads(state['data'])
                        except:
                            state['data'] = {}
                    return state
                return None

    async def set_user_state(self, user_id: int, state: str, data: Dict[str, Any] = None):
        """Set user's conversation state"""
        async with aiosqlite.connect(self.db_path) as db:
            data_json = json.dumps(data) if data else None

            await db.execute(
                """INSERT OR REPLACE INTO user_states (user_id, state, data, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                (user_id, state, data_json)
            )
            await db.commit()

    async def clear_user_state(self, user_id: int):
        """Clear user's conversation state"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM user_states WHERE user_id = ?", (user_id,))
            await db.commit()

    async def get_users_for_update(self, days_threshold: int = 30) -> List[Dict[str, Any]]:
        """Get users who need profile updates"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                    """SELECT u.*, up.answer_1, up.answer_2, up.answer_3 
                       FROM users u
                       JOIN user_profiles up ON u.id = up.user_id
                       WHERE u.is_active = 1 
                       AND datetime(u.last_profile_update) < datetime('now', '-{} days')""".format(days_threshold)
            ) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

    async def find_profiles_with_keywords(self, user_id: int, keywords: List[str],
                                          limit: int = 20) -> List[Dict[str, Any]]:
        """Find profiles that contain specific keywords"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create search condition for keywords
            keyword_conditions = []
            params = [user_id]

            for keyword in keywords[:5]:  # Limit to 5 keywords
                keyword_conditions.append("(up.answer_1 LIKE ? OR up.answer_2 LIKE ? OR up.answer_3 LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

            if not keyword_conditions:
                return []

            query = f"""
                SELECT u.telegram_id, u.username, u.first_name, u.last_name,
                       up.answer_1, up.answer_2, up.answer_3, up.keywords
                FROM user_profiles up
                JOIN users u ON up.user_id = u.id
                WHERE up.user_id != ? AND u.is_active = 1
                AND ({' OR '.join(keyword_conditions)})
                ORDER BY up.updated_at DESC
                LIMIT ?
            """
            params.append(limit)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                profiles = []
                for row in rows:
                    profile = dict(zip(columns, row))
                    # Parse keywords
                    if profile.get('keywords'):
                        try:
                            profile['keywords'] = json.loads(profile['keywords'])
                        except:
                            profile['keywords'] = []
                    profiles.append(profile)
                return profiles

    async def get_all_active_profiles(self, exclude_user_id: int = None) -> List[Dict[str, Any]]:
        """Get all active user profiles for matching"""
        async with aiosqlite.connect(self.db_path) as db:
            query = """SELECT u.telegram_id, u.username, u.first_name, u.last_name,
                              up.answer_1, up.answer_2, up.answer_3, up.user_id, up.keywords
                       FROM user_profiles up
                       JOIN users u ON up.user_id = u.id
                       WHERE u.is_active = 1"""
            params = []

            if exclude_user_id:
                query += " AND up.user_id != ?"
                params.append(exclude_user_id)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                profiles = []
                for row in rows:
                    profile = dict(zip(columns, row))
                    # Parse keywords
                    if profile.get('keywords'):
                        try:
                            profile['keywords'] = json.loads(profile['keywords'])
                        except:
                            profile['keywords'] = []
                    profiles.append(profile)
                return profiles

    async def find_similar_profiles(self, user_id: int, embedding: List[float] = None,
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar user profiles - fallback to all profiles since we don't have vector search in SQLite"""
        # Since SQLite doesn't support vector similarity, we'll use Qdrant for this
        # Import here to avoid circular imports
        try:
            from .vector_db import vector_db
            if embedding:
                return vector_db.search_similar_profiles(embedding, user_id, limit)
            else:
                # Fallback: return all profiles except user's own
                return await self.get_all_active_profiles(exclude_user_id=user_id)
        except Exception as e:
            logger.warning(f"Vector search failed, using fallback: {e}")
            # Fallback: return all profiles except user's own
            return await self.get_all_active_profiles(exclude_user_id=user_id)

    async def update_user_phone(self, user_id: int, phone: str):
        """Update user's phone number"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET phone = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (phone, user_id)
            )
            await db.commit()
            logger.info(f"Updated phone for user {user_id}")

    async def update_user_birthday(self, user_id: int, birthday: str):
        """Update user's birthday"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET birthday = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (birthday, user_id)
            )
            await db.commit()
            logger.info(f"Updated birthday for user {user_id}")

    async def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information including phone and birthday"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                    "SELECT phone, birthday FROM users WHERE id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        'phone': row[0],
                        'birthday': row[1]
                    }

    async def delete_user(self, user_id: int) -> bool:
        """Delete user and all related data"""
        async with aiosqlite.connect(self.db_path) as db:
            try:
                # Get user info before deletion
                async with db.execute(
                        "SELECT telegram_id, first_name, username FROM users WHERE id = ?",
                        (user_id,)
                ) as cursor:
                    user = await cursor.fetchone()

                if not user:
                    logger.warning(f"User {user_id} not found for deletion")
                    return False

                telegram_id, first_name, username = user
                user_display = f"{first_name or 'Unknown'}"
                if username:
                    user_display += f" (@{username})"

                # Delete user (CASCADE will handle related records)
                await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
                await db.commit()

                logger.info(f"🗑️ USER DELETED | {user_display} (ID: {telegram_id}) removed from SQLite")
                return True

            except Exception as e:
                logger.error(f"Failed to delete user {user_id}: {e}")
                return False

    async def delete_user_by_telegram_id(self, telegram_id: int) -> bool:
        """Delete user by Telegram ID"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                    "SELECT id FROM users WHERE telegram_id = ?", (telegram_id,)
            ) as cursor:
                user = await cursor.fetchone()

            if user:
                return await self.delete_user(user[0])
            else:
                logger.warning(f"User with Telegram ID {telegram_id} not found")
                return False


# Global database instance
db = Database()
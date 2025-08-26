from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger
from config import (
    TELEGRAM_BOT_TOKEN, BOT_CONFIG, QUESTIONS, VALIDATION_PATTERNS,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from .database import db
from .embeddings import embedding_service
from .llm_service import llm_service
from .text_processing import text_processor
from .vector_db import vector_db


class ProfileStates(StatesGroup):
    waiting_answer_1 = State()
    waiting_answer_2 = State()
    waiting_answer_3 = State()
    waiting_birthday = State()
    waiting_phone = State()


class BusinessMatchingBot:
    def __init__(self):
        self.bot = None
        self.dp = None
        self.router = Router()
        self.processing_users = set()  # Track users currently being processed
        self.setup_handlers()

    def _validate_birthday(self, birthday_text: str) -> bool:
        """Validate birthday format"""
        import re
        from datetime import datetime

        patterns = VALIDATION_PATTERNS["birthday"]

        if not any(re.match(pattern, birthday_text) for pattern in patterns):
            return False

        try:
            # Пробуем парсить дату
            for fmt in ['%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y']:
                try:
                    date_obj = datetime.strptime(birthday_text, fmt)
                    # Проверяем разумные границы (от 16 до 100 лет)
                    current_year = datetime.now().year
                    if 1924 <= date_obj.year <= current_year - 16:
                        return True
                except ValueError:
                    continue
            return False
        except:
            return False

    def _validate_phone(self, phone_text: str) -> bool:
        """Validate phone number format"""
        import re

        # Убираем все кроме цифр и +
        clean_phone = re.sub(r'[^\d+]', '', phone_text)

        patterns = VALIDATION_PATTERNS["phone"]

        return any(re.match(pattern, clean_phone) for pattern in patterns)

    def setup_handlers(self):
        """Setup message handlers"""
        # Command handlers
        self.router.message.register(self.start_command, Command("start"))
        self.router.message.register(self.match_command, Command("match"))

        # Menu button handlers
        self.router.message.register(
            self.find_matches_handler,
            F.text == "🔍 Подобрать участников"
        )
        self.router.message.register(
            self.update_profile_handler,
            F.text == "📝 Обновить анкету"
        )
        self.router.message.register(
            self.show_profile_handler,
            F.text == "👤 Мой профиль"
        )

        # State handlers for profile creation
        self.router.message.register(
            self.handle_answer_1,
            StateFilter(ProfileStates.waiting_answer_1)
        )
        self.router.message.register(
            self.handle_answer_2,
            StateFilter(ProfileStates.waiting_answer_2)
        )
        self.router.message.register(
            self.handle_answer_3,
            StateFilter(ProfileStates.waiting_answer_3)
        )
        self.router.message.register(
            self.handle_birthday,
            StateFilter(ProfileStates.waiting_birthday)
        )
        self.router.message.register(
            self.handle_phone,
            StateFilter(ProfileStates.waiting_phone)
        )

    async def start_command(self, message: Message, state: FSMContext):
        """Handle /start command"""
        user = message.from_user

        # Prevent concurrent processing of the same user
        if user.id in self.processing_users:
            logger.info(f"🚫 CONCURRENT START | User {user.id} already being processed, ignoring")
            return

        self.processing_users.add(user.id)

        try:
            await self._handle_start_command(message, state, user)
        finally:
            self.processing_users.discard(user.id)

    async def _handle_start_command(self, message: Message, state: FSMContext, user):
        """Internal start command handler"""
        user_display = f"{user.first_name or 'Unknown'}"
        if user.username:
            user_display += f" (@{user.username})"
        logger.info(f"🚀 START COMMAND | {user_display} (ID: {user.id}) started the bot")

        try:
            # Get or create user in database
            db_user = await db.get_or_create_user(
                telegram_id=user.id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name
            )

            # Check if user already has a profile
            profile = await db.get_user_profile(db_user['id'])

            if profile:
                logger.info(f"👤 EXISTING USER | {user_display} already has a profile, showing main menu")
                # User already has profile
                keyboard = ReplyKeyboardMarkup(
                    keyboard=[
                        [KeyboardButton(text="🔍 Подобрать участников")],
                        [KeyboardButton(text="📝 Обновить анкету")],
                        [KeyboardButton(text="👤 Мой профиль")]
                    ],
                    resize_keyboard=True
                )

                await message.answer(
                    f"Добро пожаловать обратно, {user.first_name}! 👋\n\n"
                    "Ваш профиль уже создан. Выберите действие:",
                    reply_markup=keyboard
                )
            else:
                logger.info(f"🆕 NEW USER | Starting onboarding for {user_display}")
                # New user - start onboarding
                await self.start_onboarding(message, state, db_user['id'], user_display)

        except Exception as e:
            logger.error(f"❌ START COMMAND ERROR | {user_display}: {e}")
            await message.answer(
                "Произошла ошибка при обработке команды. Попробуйте еще раз."
            )

    async def start_onboarding(self, message: Message, state: FSMContext, user_id: int, user_display: str = None):
        """Start the onboarding process"""
        if not user_display:
            user = message.from_user
            user_display = f"{user.first_name or 'Unknown'}"
            if user.username:
                user_display += f" (@{user.username})"
        logger.info(f"🆕 ONBOARDING START | {user_display}")
        await message.answer(
            "Добро пожаловать в наше бизнес-сообщество! 🚀\n\n"
            "Для участия в сообществе нужно заполнить короткую анкету из 3 вопросов. "
            "Это поможет находить для вас наиболее подходящих собеседников.\n\n"
            "Начнем?"
        )

        # Set state and ask first question
        await state.set_state(ProfileStates.waiting_answer_1)
        await state.update_data(user_id=user_id, user_display=user_display)
        await message.answer(f"**Вопрос 1 из 3:**\n{QUESTIONS[1]}")

    async def handle_answer_1(self, message: Message, state: FSMContext):
        """Handle first answer"""
        data = await state.get_data()
        user_display = data.get('user_display', f"User {message.from_user.id}")

        if len(message.text) > BOT_CONFIG["max_answer_length"]:
            await message.answer(ERROR_MESSAGES["answer_too_long"].format(
                max_length=BOT_CONFIG["max_answer_length"]
            ))
            return

        if len(message.text) < BOT_CONFIG["min_answer_length"]:
            await message.answer(ERROR_MESSAGES["answer_too_short"].format(
                min_length=BOT_CONFIG["min_answer_length"]
            ))
            return

        logger.info(f"📝 ANSWER 1 | {user_display} answered: {message.text[:50]}...")
        await state.update_data(answer_1=message.text)
        await state.set_state(ProfileStates.waiting_answer_2)
        await message.answer(f"**Вопрос 2 из 3:**\n{QUESTIONS[2]}")

    async def handle_answer_2(self, message: Message, state: FSMContext):
        """Handle second answer"""
        data = await state.get_data()
        user_display = data.get('user_display', f"User {message.from_user.id}")

        if len(message.text) > BOT_CONFIG["max_answer_length"]:
            await message.answer(ERROR_MESSAGES["answer_too_long"].format(
                max_length=BOT_CONFIG["max_answer_length"]
            ))
            return

        if len(message.text) < BOT_CONFIG["min_answer_length"]:
            await message.answer(ERROR_MESSAGES["answer_too_short"].format(
                min_length=BOT_CONFIG["min_answer_length"]
            ))
            return

        logger.info(f"📝 ANSWER 2 | {user_display} answered: {message.text[:50]}...")
        await state.update_data(answer_2=message.text)
        await state.set_state(ProfileStates.waiting_answer_3)
        await message.answer(f"**Вопрос 3 из 3:**\n{QUESTIONS[3]}")

    async def handle_answer_3(self, message: Message, state: FSMContext):
        """Handle third answer and complete profile"""
        data = await state.get_data()
        user_display = data.get('user_display', f"User {message.from_user.id}")

        if len(message.text) > BOT_CONFIG["max_answer_length"]:
            await message.answer(ERROR_MESSAGES["answer_too_long"].format(
                max_length=BOT_CONFIG["max_answer_length"]
            ))
            return

        if len(message.text) < BOT_CONFIG["min_answer_length"]:
            await message.answer(ERROR_MESSAGES["answer_too_short"].format(
                min_length=BOT_CONFIG["min_answer_length"]
            ))
            return

        logger.info(f"📝 ANSWER 3 | {user_display} answered: {message.text[:50]}...")
        await state.update_data(answer_3=message.text)
        await state.set_state(ProfileStates.waiting_birthday)
        await message.answer(f"**Вопрос 4 из 5:**\n{QUESTIONS[4]}")

    async def handle_birthday(self, message: Message, state: FSMContext):
        """Handle birthday input"""
        data = await state.get_data()
        user_display = data.get('user_display', f"User {message.from_user.id}")

        # Простая валидация даты
        birthday_text = message.text.strip()
        if not self._validate_birthday(birthday_text):
            await message.answer(
                ERROR_MESSAGES["invalid_birthday"]
            )
            return

        logger.info(f"🎂 BIRTHDAY | {user_display} provided birthday: {birthday_text}")
        await state.update_data(birthday=birthday_text)
        await state.set_state(ProfileStates.waiting_phone)

        # Create keyboard with phone request and skip option
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="📱 Поделиться номером", request_contact=True)],
            ],
            resize_keyboard=True,
            one_time_keyboard=True
        )

        await message.answer(
            f"**Вопрос 5 из 5:**\n{QUESTIONS[5]}",
            reply_markup=keyboard
        )

    async def handle_phone(self, message: Message, state: FSMContext):
        """Handle phone number or skip"""
        data = await state.get_data()
        user_id = data['user_id']
        user_display = data.get('user_display', f"User {user_id}")

        # Get phone number if provided
        phone = None
        if message.contact:
            phone = message.contact.phone_number
            logger.info(f"📱 PHONE CONTACT | {user_display} provided phone: {phone}")
        elif message.text and message.text != "📱 Поделиться номером":
            # User typed phone manually
            phone_text = message.text.strip()
            if self._validate_phone(phone_text):
                phone = phone_text
                logger.info(f"📱 PHONE TYPED | {user_display} typed phone: {phone}")
            else:
                await message.answer(
                    ERROR_MESSAGES["invalid_phone"]
                )
                return
        else:
            await message.answer(
                ERROR_MESSAGES["phone_required"]
            )
            return

        answers = {
            'answer_1': data['answer_1'],
            'answer_2': data['answer_2'],
            'answer_3': data['answer_3'],
            'birthday': data['birthday'],
            'phone': phone
        }

    async def complete_profile(self, message: Message, state: FSMContext,
                               user_id: int, answers: Dict[str, str], user_display: str = None):
        """Complete profile creation/update"""
        if not user_display:
            # Try to get user display name from message
            user = message.from_user
            user_display = f"{user.first_name or 'Unknown'}"
            if user.username:
                user_display += f" (@{user.username})"
            user_display += f" (ID: {user.id})"
        logger.info(f"✅ PROFILE COMPLETION START | {user_display}")
        try:
            # Update user with phone number and birthday
            await db.update_user_phone(user_id, answers['phone'])
            await db.update_user_birthday(user_id, answers['birthday'])
            logger.debug(f"📱 PHONE UPDATED | {user_display}: {answers['phone']}")
            logger.debug(f"🎂 BIRTHDAY UPDATED | {user_display}: {answers['birthday']}")

            # Process text and create embedding
            processed_data = text_processor.prepare_profile_text(
                answers['answer_1'],
                answers['answer_2'],
                answers['answer_3']
            )
            logger.debug(f"🔤 TEXT PROCESSED | {user_display}, keywords: {processed_data['keywords'][:5]}")

            # Create embedding from processed text
            try:
                embedding = embedding_service.create_profile_embedding(
                    processed_data['clean_answers']['answer_1'],
                    processed_data['clean_answers']['answer_2'],
                    processed_data['clean_answers']['answer_3']
                )
                logger.info(f"🧠 EMBEDDING CREATED | {user_display}, dimension: {len(embedding)}")
            except Exception as e:
                logger.warning(f"⚠️ EMBEDDING FAILED | {user_display}: {e}")
                embedding = None

            # Save profile to database
            await db.save_user_profile(
                user_id,
                processed_data['clean_answers']['answer_1'],
                processed_data['clean_answers']['answer_2'],
                processed_data['clean_answers']['answer_3'],
                embedding,
                processed_data['keywords']
            )
            logger.info(f"💾 PROFILE SAVED TO SQLITE | {user_display}")

            # Save to vector database if embedding was created
            if embedding:
                try:
                    # Get user info for vector DB
                    user_info = await db.get_or_create_user(message.from_user.id)
                    profile_data = {
                        "telegram_id": user_info['telegram_id'],
                        "username": user_info['username'],
                        "first_name": user_info['first_name'],
                        "last_name": user_info['last_name'],
                        "answer_1": processed_data['clean_answers']['answer_1'],
                        "answer_2": processed_data['clean_answers']['answer_2'],
                        "answer_3": processed_data['clean_answers']['answer_3'],
                        "keywords": processed_data['keywords']
                    }
                    vector_db.save_profile_embedding(user_id, embedding, profile_data)
                    logger.info(f"🔍 PROFILE SAVED TO QDRANT | {user_display}")
                except Exception as e:
                    logger.warning(f"⚠️ QDRANT SAVE FAILED | {user_display}: {e}")

            # Clear state
            await state.clear()

            logger.success(f"🎉 PROFILE COMPLETION SUCCESS | {user_display}")

            # Create main menu
            keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="🔍 Подобрать участников")],
                    [KeyboardButton(text="📝 Обновить анкету")],
                    [KeyboardButton(text="👤 Мой профиль")]
                ],
                resize_keyboard=True
            )

            await message.answer(
                "Спасибо! Ваш профиль создан! ✅\n\n"
                "Теперь вы можете:\n"
                "• Найти подходящих участников для знакомства\n"
                "• Обновить свою анкету в любое время\n\n"
                "Мы будем напоминать обновить профиль раз в месяц, "
                "чтобы рекомендации оставались актуальными.",
                reply_markup=keyboard
            )

        except Exception as e:
            logger.error(f"❌ PROFILE COMPLETION ERROR | {user_display}: {e}")
            await message.answer(
                "Произошла ошибка при сохранении профиля. Попробуйте еще раз."
            )

    async def update_profile_handler(self, message: Message, state: FSMContext):
        """Start profile update process"""
        user_display = f"{message.from_user.first_name or 'Unknown'}"
        if message.from_user.username:
            user_display += f" (@{message.from_user.username})"
        logger.info(f"📝 UPDATE PROFILE START | {user_display} (ID: {message.from_user.id})")
        user = message.from_user
        db_user = await db.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )

        logger.debug(f"📝 UPDATE PROFILE | Starting update flow for {user_display}")

        await message.answer(
            "Давайте обновим вашу анкету! 📝\n\n"
            "Ответьте на те же 3 вопроса. Если что-то не изменилось, "
            "можете повторить предыдущий ответ."
        )

        await state.set_state(ProfileStates.waiting_answer_1)
        await state.update_data(user_id=db_user['id'], user_display=user_display)
        await message.answer(f"**Вопрос 1 из 3:**\n{QUESTIONS[1]}")

    async def show_profile_handler(self, message: Message):
        """Show user's current profile"""
        user_display = f"{message.from_user.first_name or 'Unknown'}"
        if message.from_user.username:
            user_display += f" (@{message.from_user.username})"
        logger.info(f"👤 SHOW PROFILE | {user_display} (ID: {message.from_user.id}) requested profile")
        user = message.from_user
        db_user = await db.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )

        profile = await db.get_user_profile(db_user['id'])
        user_info = await db.get_user_info(db_user['id'])

        if not profile:
            logger.warning(f"👤 PROFILE NOT FOUND | {user_display} has no profile")
            await message.answer("У вас пока нет профиля. Используйте /start для создания.")
            return

        logger.debug(f"👤 PROFILE DISPLAY | {user_display}: {profile['answer_1'][:30]}...")

        # Handle date formatting - SQLite returns string, not datetime object
        try:
            if isinstance(profile['updated_at'], str):
                # Parse SQLite datetime string
                from datetime import datetime
                updated_at = datetime.fromisoformat(profile['updated_at'].replace('Z', '+00:00'))
                formatted_date = updated_at.strftime('%d.%m.%Y')
            else:
                # Already datetime object
                formatted_date = profile['updated_at'].strftime('%d.%m.%Y')
        except (ValueError, KeyError, TypeError):
            formatted_date = "Неизвестно"

        # Escape text for MarkdownV2
        safe_answer_1 = self._escape_markdown(profile['answer_1'])
        safe_answer_2 = self._escape_markdown(profile['answer_2'])
        safe_answer_3 = self._escape_markdown(profile['answer_3'])
        safe_date = self._escape_markdown(formatted_date)
        safe_birthday = self._escape_markdown(user_info.get('birthday', 'Не указан'))
        safe_phone = self._escape_markdown(user_info.get('phone', 'Не указан'))

        profile_text = f"""*Ваш профиль:*

*🏢 Сфера деятельности:*
{safe_answer_1}

*🔍 Что ищете в сообществе:*
{safe_answer_2}

*🤝 Чем можете помочь:*
{safe_answer_3}

*🎂 День рождения:* {safe_birthday}
*📱 Телефон:* {safe_phone}

_Последнее обновление: {safe_date}_
        """

        try:
            await message.answer(profile_text, parse_mode="MarkdownV2")
            logger.success(f"👤 PROFILE DISPLAYED | {user_display}: profile shown successfully")
        except Exception as parse_error:
            logger.warning(f"⚠️ PROFILE MARKDOWN ERROR | {user_display}: {parse_error}")
            # Fallback: send without formatting
            plain_text = self._strip_markdown(profile_text)
            await message.answer(plain_text)
            logger.info(f"📤 PROFILE PLAIN TEXT | {user_display}: fallback message sent")

    async def find_matches_handler(self, message: Message):
        """Find matching participants"""
        user_display = f"{message.from_user.first_name or 'Unknown'}"
        if message.from_user.username:
            user_display += f" (@{message.from_user.username})"
        logger.info(f"🔍 FIND MATCHES START | {user_display} (ID: {message.from_user.id}) clicked find matches")
        user = message.from_user
        db_user = await db.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )

        await self.find_matches(message, db_user['id'], user_display)

    async def match_command(self, message: Message):
        """Handle /match command"""
        user_display = f"{message.from_user.first_name or 'Unknown'}"
        if message.from_user.username:
            user_display += f" (@{message.from_user.username})"
        logger.info(f"🔍 MATCH COMMAND | {user_display} (ID: {message.from_user.id}) used /match command")
        user = message.from_user
        db_user = await db.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )

        await self.find_matches(message, db_user['id'], user_display)

    async def find_matches(self, message: Message, user_id: int, user_display: str = None):
        """Find matching participants"""
        if not user_display:
            user_display = f"User {user_id}"
        logger.info(f"🔍 MATCHING PROCESS START | {user_display}")

        # Check if user has profile
        profile = await db.get_user_profile(user_id)
        if not profile:
            logger.warning(f"🔍 NO PROFILE | {user_display} tried to find matches without profile")
            await message.answer(
                "Сначала создайте профиль с помощью команды /start"
            )
            return

        logger.debug(f"🔍 USER PROFILE | {user_display}: {profile['answer_1'][:50]}...")
        await message.answer("Ищу для вас наиболее подходящих собеседников... 🔍")

        try:
            # First try keyword-based filtering for better initial results
            keywords = profile.get('keywords', [])
            candidate_profiles = []

            if keywords:
                # Get profiles with matching keywords
                keyword_profiles = await db.find_profiles_with_keywords(
                    user_id, keywords, limit=15
                )
                logger.info(
                    f"🔍 KEYWORD SEARCH | {user_display}: found {len(keyword_profiles)} profiles with keywords {keywords[:3]}")
                candidate_profiles.extend(keyword_profiles)
            else:
                logger.debug(f"🔍 NO KEYWORDS | {user_display} has no keywords for search")

            # Add vector similarity results
            try:
                # Try to get embedding from Qdrant or create new one
                user_embedding = None

                # First, try to get existing embedding from Qdrant
                try:
                    from .vector_db import vector_db

                    # Try to get user's embedding
                    user_embedding = vector_db.get_user_embedding(user_id)

                    if user_embedding:
                        logger.info(f"🔍 EXISTING EMBEDDING | {user_display}: found in Qdrant")
                        vector_profiles = vector_db.search_similar_profiles(user_embedding, user_id, limit=15)
                        logger.info(f"🔍 VECTOR SEARCH | {user_display}: found {len(vector_profiles)} similar profiles")
                    else:
                        logger.warning(f"🔍 NO QDRANT PROFILE | {user_display}: creating embedding")
                        # Create embedding for this user
                        processed_data = text_processor.prepare_profile_text(
                            profile['answer_1'], profile['answer_2'], profile['answer_3']
                        )
                        user_embedding = embedding_service.create_profile_embedding(
                            processed_data['clean_answers']['answer_1'],
                            processed_data['clean_answers']['answer_2'],
                            processed_data['clean_answers']['answer_3']
                        )

                        # Save to Qdrant
                        user_info = await db.get_or_create_user(message.from_user.id)
                        profile_data = {
                            "telegram_id": user_info['telegram_id'],
                            "username": user_info['username'],
                            "first_name": user_info['first_name'],
                            "last_name": user_info['last_name'],
                            "answer_1": profile['answer_1'],
                            "answer_2": profile['answer_2'],
                            "answer_3": profile['answer_3'],
                            "keywords": profile.get('keywords', [])
                        }
                        vector_db.save_profile_embedding(user_id, user_embedding, profile_data)
                        logger.info(f"🔍 EMBEDDING CREATED & SAVED | {user_display}")

                        # Now do vector search
                        vector_profiles = vector_db.search_similar_profiles(user_embedding, user_id, limit=15)
                        logger.info(f"🔍 VECTOR SEARCH | {user_display}: found {len(vector_profiles)} similar profiles")

                except Exception as qdrant_error:
                    logger.warning(f"🔍 QDRANT ERROR | {user_display}: {qdrant_error}")
                    # Fallback to SQLite search
                    vector_profiles = await db.get_all_active_profiles(exclude_user_id=user_id)
                    logger.info(
                        f"🔍 FALLBACK SEARCH | {user_display}: found {len(vector_profiles)} profiles (Qdrant failed)")

            except Exception as e:
                logger.warning(f"🔍 VECTOR SEARCH FAILED | {user_display}: {e}")
                vector_profiles = []

            # Combine and deduplicate
            seen_users = set()
            combined_profiles = []

            for profiles_list in [candidate_profiles, vector_profiles]:
                for p in profiles_list:
                    user_key = p['telegram_id']
                    if user_key not in seen_users:
                        seen_users.add(user_key)
                        combined_profiles.append(p)

            similar_profiles = combined_profiles[:20]  # Limit to top 20 for LLM analysis

            logger.info(f"🔍 CANDIDATES COMBINED | {user_display}: {len(similar_profiles)} total candidates")

            if not similar_profiles:
                logger.warning(f"🔍 NO CANDIDATES | {user_display}: no similar profiles found")
                await message.answer(
                    "Пока не найдено подходящих участников. "
                    "Попробуйте позже, когда в сообществе будет больше людей."
                )
                return

            # Log candidate details
            for i, candidate in enumerate(similar_profiles[:5], 1):
                logger.debug(
                    f"🔍 CANDIDATE {i} | {candidate.get('first_name', 'Unknown')}: {candidate.get('answer_1', '')[:30]}...")

            # Use LLM to find best matches
            logger.info(f"🤖 LLM ANALYSIS START | {user_display}: analyzing {len(similar_profiles)} candidates")
            best_matches = await llm_service.find_best_matches(
                profile,
                similar_profiles,
                top_k=5
            )

            logger.info(f"🤖 LLM ANALYSIS COMPLETE | {user_display}: returned {len(best_matches)} best matches")

            # Log match details
            for i, match in enumerate(best_matches, 1):
                score = match.get('match_score', 'N/A')
                name = match.get('first_name', 'Unknown')
                reason = match.get('match_reason', 'No reason')[:50]
                logger.debug(f"🤖 MATCH {i} | {name} (Score: {score}): {reason}...")

            if not best_matches:
                logger.warning(f"🤖 NO MATCHES | {user_display}: LLM returned no matches")
                await message.answer(
                    "Не удалось найти подходящих участников. Попробуйте позже."
                )
                return

            # Generate summary message
            logger.debug(f"🤖 SUMMARY GENERATION | {user_display}: generating summary for {len(best_matches)} matches")
            summary = await llm_service.generate_match_summary(profile, best_matches)

            # Format matches message
            # Escape summary text
            safe_summary = self._escape_markdown(summary)
            matches_text = f"{safe_summary}\n\n*Рекомендованные контакты:*\n\n"

            for i, match in enumerate(best_matches, 1):
                name = match.get('first_name', '') or match.get('username', f'Участник {i}')
                username_text = f"@{match['username']}" if match.get('username') else "Нет username"

                # Escape special characters for Markdown
                safe_name = self._escape_markdown(name)
                safe_username = self._escape_markdown(username_text)
                safe_answer_1 = self._escape_markdown(match['answer_1'])
                safe_answer_2 = self._escape_markdown(match['answer_2'])
                safe_answer_3 = self._escape_markdown(match['answer_3'])
                safe_reason = self._escape_markdown(match.get('match_reason', 'Схожие интересы'))

                matches_text += f"""*{i}\\. {safe_name}* \\({safe_username}\\)
*Сфера:* {safe_answer_1}
*Ищет:* {safe_answer_2}
*Может помочь:* {safe_answer_3}
*Почему подходит:* {safe_reason}

"""

            matches_text += "Удачного знакомства\\! 🤝"

            try:
                # Try to send with MarkdownV2 first
                await message.answer(matches_text, parse_mode="MarkdownV2")
                logger.success(f"🎉 MATCHING SUCCESS | {user_display}: sent {len(best_matches)} matches")
            except Exception as parse_error:
                logger.warning(f"⚠️ MARKDOWN PARSE ERROR | {user_display}: {parse_error}")
                # Fallback: create safe plain text message
                plain_text = self._create_safe_markdown_message(summary, best_matches)
                await message.answer(plain_text)
                logger.info(f"📤 SENT PLAIN TEXT | {user_display}: fallback message sent")


        except Exception as e:
            logger.error(f"❌ MATCHING ERROR | {user_display}: {e}")
            await message.answer(
                "Произошла ошибка при поиске участников. Попробуйте позже."
            )

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for MarkdownV2"""
        if not text:
            return ""

        # Characters that need to be escaped in MarkdownV2 (order matters!)
        special_chars = ['\\', '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']

        for char in special_chars:
            if char == '\\':
                text = text.replace(char, '\\\\')
            else:
                text = text.replace(char, f'\\{char}')

        return text

    def _strip_markdown(self, text: str) -> str:
        """Remove all markdown formatting"""
        import re
        # Remove markdown formatting
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove *text*
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Remove _text_
        text = re.sub(r'\\(.)', r'\1', text)  # Remove escape characters
        return text

    def _create_safe_markdown_message(self, summary: str, matches: List[Dict[str, Any]]) -> str:
        """Create a safely formatted markdown message"""
        try:
            # Start with plain text version as fallback
            plain_parts = [summary, "\n\nРекомендованные контакты:\n"]

            for i, match in enumerate(matches, 1):
                name = match.get('first_name', '') or match.get('username', f'Участник {i}')
                username_text = f"@{match['username']}" if match.get('username') else "Нет username"

                plain_parts.append(f"""
{i}. {name} ({username_text})
Сфера: {match['answer_1']}
Ищет: {match['answer_2']}
Может помочь: {match['answer_3']}
Почему подходит: {match.get('match_reason', 'Схожие интересы')}
""")

            plain_parts.append("\nУдачного знакомства! 🤝")
            return "".join(plain_parts)

        except Exception as e:
            logger.error(f"Failed to create safe markdown message: {e}")
            return "Найдены подходящие участники для знакомства! 🤝"

    async def setup_bot(self):
        """Setup the bot"""
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

        # Create bot and dispatcher
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.dp = Dispatcher(storage=MemoryStorage())

        # Include router
        self.dp.include_router(self.router)

        return self.bot, self.dp


# Global bot instance
bot_instance = BusinessMatchingBot()
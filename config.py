import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys (loaded from .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_PATH = os.getenv("DATABASE_PATH", "business_bot.db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", 78481301))
UPDATE_INTERVAL_DAYS = int(os.getenv("UPDATE_INTERVAL_DAYS", 30))

# LLM Configuration
LLM_CONFIG = {
    "top_k": 15,
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_provider": "deepseek",
    "deepseek_model": "deepseek-chat",
    "deepseek_base_url": "https://api.deepseek.com",
    "max_tokens": 700,
    "temperature": 0.7
}

# Text Processing Configuration
TEXT_SPLIT_PARAMS = {
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "min_chunk_size": 100,
    "use_smart_splitting": True
}

# Bot Configuration
BOT_CONFIG = {
    "max_answer_length": 2000,
    "min_answer_length": 10,
    "profile_questions_count": 3,
    "matching_candidates_limit": 20,
    "vector_search_limit": 15,
    "keyword_search_limit": 15
}

# Database Configuration
DB_CONFIG = {
    "vector_dimension": 384,
    "collection_name": "user_profiles",
    "connection_timeout": 30,
    "max_retries": 3
}

# System Prompts
SYSTEM_PROMPT = """Ты - эксперт по бизнес-нетворкингу и подбору деловых контактов. 
Твоя задача - анализировать профили участников бизнес-сообщества и находить наиболее подходящих людей для знакомства и сотрудничества.

Ты работаешь с русскоязычным сообществом предпринимателей, инвесторов, специалистов и экспертов.

Принципы подбора:
1. Взаимодополняющие навыки и потребности
2. Общие интересы и сферы деятельности  
3. Потенциал для взаимовыгодного сотрудничества
4. Совпадение того, что один ищет, с тем, что другой может предложить

Всегда отвечай на русском языке, будь дружелюбным и профессиональным."""

CONTEXT_PROMPT_TEMPLATE = """
Профиль пользователя:
- Сфера деятельности: {user_answer_1}
- Что ищет в сообществе: {user_answer_2}
- Чем может помочь: {user_answer_3}

Кандидаты для сопоставления:
{candidates_text}

Проанализируй профили и выбери {top_k} наиболее подходящих кандидатов для этого пользователя.

Критерии отбора:
1. Взаимодополняющие навыки и потребности
2. Общие интересы и сферы деятельности
3. Потенциал для взаимовыгодного сотрудничества
4. Совпадение того, что один ищет, с тем, что другой может предложить

Верни результат в формате JSON:
{{
    "matches": [
        {{
            "candidate_index": номер_кандидата_из_списка,
            "match_score": оценка_от_1_до_10,
            "reason": "краткое_объяснение_почему_подходит"
        }}
    ]
}}

Сортируй по убыванию match_score.
"""

SUMMARY_PROMPT_TEMPLATE = """
Создай краткое и дружелюбное сообщение для пользователя с результатами подбора контактов.

Профиль пользователя:
- Сфера деятельности: {user_answer_1}
- Что ищет: {user_answer_2}
- Чем может помочь: {user_answer_3}

Найденные контакты:
{matches_text}

Сообщение должно:
1. Быть дружелюбным и мотивирующим
2. Кратко объяснить, почему эти люди подходят
3. Поощрить к знакомству
4. Быть не длиннее 500 символов

Не используй формат JSON, просто напиши текст сообщения.
"""

# Questions Configuration
QUESTIONS = {
    1: "Расскажите о вашей сфере деятельности?",
    2: "Что вы ищете в сообществе?",
    3: "Чем можете помочь другим участникам?",
    4: "Укажите вашу дату рождения (например: 15.03.1990)?",
    5: "Поделитесь номером телефона для связи?"
}

# Validation patterns
VALIDATION_PATTERNS = {
    "birthday": [
        r'^\d{1,2}\.\d{1,2}\.\d{4}$',
        r'^\d{1,2}/\d{1,2}/\d{4}$',
        r'^\d{1,2}-\d{1,2}-\d{4}$'
    ],
    "phone": [
        r'^\+\d{10,15}$',  # +1234567890
        r'^\d{10,11}$',  # 1234567890 или 81234567890
    ]
}

# Error messages
ERROR_MESSAGES = {
    "answer_too_long": "Ответ слишком длинный. Пожалуйста, сократите до {max_length} символов.",
    "answer_too_short": "Ответ слишком короткий. Пожалуйста, напишите минимум {min_length} символов.",
    "invalid_birthday": "Пожалуйста, укажите дату в правильном формате (например: 15.03.1990 или 15/03/1990)",
    "invalid_phone": "Пожалуйста, укажите корректный номер телефона или воспользуйтесь кнопкой 'Поделиться номером'",
    "phone_required": "Номер телефона обязателен. Пожалуйста, поделитесь номером или введите его вручную.",
    "profile_save_error": "Произошла ошибка при сохранении профиля. Попробуйте еще раз.",
    "matching_error": "Произошла ошибка при поиске участников. Попробуйте позже.",
    "no_profile": "Сначала создайте профиль с помощью команды /start",
    "no_matches": "Пока не найдено подходящих участников. Попробуйте позже, когда в сообществе будет больше людей."
}

# Success messages
SUCCESS_MESSAGES = {
    "profile_created": """Спасибо! Ваш профиль создан! ✅

Теперь вы можете:
• Найти подходящих участников для знакомства
• Обновить свою анкету в любое время

Мы будем напоминать обновить профиль раз в месяц, чтобы рекомендации оставались актуальными.""",

    "welcome_back": "Добро пожаловать обратно, {first_name}! 👋\n\nВаш профиль уже создан. Выберите действие:",

    "onboarding_start": """Добро пожаловать в наше бизнес-сообщество! 🚀

Для участия в сообществе нужно заполнить короткую анкету из 5 вопросов. 
Это поможет находить для вас наиболее подходящих собеседников.

Начнем?""",

    "profile_update_start": """Давайте обновим вашу анкету! 📝

Ответьте на те же вопросы. Если что-то не изменилось, можете повторить предыдущий ответ.""",

    "searching_matches": "Ищу для вас наиболее подходящих собеседников... 🔍"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "file_format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    "rotation": "1 day",
    "retention": "7 days",
    "compression": "zip"
}
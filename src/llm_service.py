import openai
from typing import List, Dict, Any
import json
from loguru import logger
from config import (
    DEEPSEEK_API_KEY, LLM_CONFIG, SYSTEM_PROMPT,
    CONTEXT_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE
)


class DeepSeekService:
    def __init__(self):
        """Initialize DeepSeek service"""
        # Set OpenAI configuration for older version
        openai.api_key = DEEPSEEK_API_KEY
        openai.api_base = LLM_CONFIG["deepseek_base_url"]

    async def find_best_matches(self, user_profile: Dict[str, Any],
                                candidate_profiles: List[Dict[str, Any]],
                                top_k: int = None) -> List[Dict[str, Any]]:
        """Use DeepSeek to find best matching profiles"""

        if top_k is None:
            top_k = LLM_CONFIG["top_k"]

        # Prepare candidate profiles text
        candidates_text = "Кандидаты для сопоставления:\n"
        for i, profile in enumerate(candidate_profiles):
            name = profile.get('first_name', '') or profile.get('username', f'Пользователь {i + 1}')
            candidates_text += f"""
            {i + 1}. {name}:
            - Сфера деятельности: {profile['answer_1']}
            - Что ищет в сообществе: {profile['answer_2']}
            - Чем может помочь: {profile['answer_3']}
            """

        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            user_answer_1=user_profile['answer_1'],
            user_answer_2=user_profile['answer_2'],
            user_answer_3=user_profile['answer_3'],
            candidates_text=candidates_text,
            top_k=min(top_k, len(candidate_profiles))
        )

        try:
            response = openai.ChatCompletion.create(
                model=LLM_CONFIG["deepseek_model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )

            result_text = response.choices[0].message['content'].strip()

            # Clean up JSON response - remove markdown code blocks
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()

            # Clean up JSON response - remove markdown code blocks
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()

            # Parse JSON response
            try:
                result = json.loads(result_text)
                matches = []

                for match in result.get('matches', []):
                    candidate_idx = match.get('candidate_index', 1) - 1  # Convert to 0-based index
                    if 0 <= candidate_idx < len(candidate_profiles):
                        profile = candidate_profiles[candidate_idx].copy()
                        profile['match_score'] = match.get('match_score', 0)
                        profile['match_reason'] = match.get('reason', 'Подходящий кандидат')
                        matches.append(profile)

                return matches[:top_k]

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response text: {result_text}")
                # Fallback: return first few candidates
                return candidate_profiles[:top_k]

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            # Fallback: return first few candidates
            return candidate_profiles[:top_k]

    async def generate_match_summary(self, user_profile: Dict[str, Any],
                                     matches: List[Dict[str, Any]]) -> str:
        """Generate a summary message for the matches"""

        matches_text = "Найденные контакты:\n"
        for i, match in enumerate(matches):
            name = match.get('first_name', '') or match.get('username', f'Участник {i + 1}')
            matches_text += f"""
            {i + 1}. {name}:
            - Сфера: {match['answer_1']}
            - Ищет: {match['answer_2']}
            - Может помочь: {match['answer_3']}
            - Причина совпадения: {match.get('match_reason', 'Подходящий профиль')}
            """

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            user_answer_1=user_profile['answer_1'],
            user_answer_2=user_profile['answer_2'],
            user_answer_3=user_profile['answer_3'],
            matches_text=matches_text
        )

        try:
            response = openai.ChatCompletion.create(
                model=LLM_CONFIG["deepseek_model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=300
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            logger.error(f"Failed to generate match summary: {e}")
            return "Вот подходящие контакты для знакомства! 🤝"


# Global LLM service instance
llm_service = DeepSeekService()
import re
from typing import List, Dict, Any
from loguru import logger
from config import TEXT_SPLIT_PARAMS


class TextProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize text processor

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or TEXT_SPLIT_PARAMS["chunk_size"]
        self.chunk_overlap = chunk_overlap or TEXT_SPLIT_PARAMS["chunk_overlap"]
        self.min_chunk_size = TEXT_SPLIT_PARAMS["min_chunk_size"]
        self.use_smart_splitting = TEXT_SPLIT_PARAMS["use_smart_splitting"]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-()«»""]', '', text)

        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting for Russian text
        sentences = re.split(r'[.!?]+\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        sentences = self.split_into_sentences(text)

        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(
                        current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    # Single sentence is too long, split it
                    if len(sentence) > self.chunk_size:
                        # Split long sentence into smaller parts
                        words = sentence.split()
                        temp_chunk = ""

                        for word in words:
                            if len(temp_chunk) + len(word) + 1 > self.chunk_size:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                temp_chunk += " " + word if temp_chunk else word

                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text for filtering"""
        if not text:
            return []

        # Simple keyword extraction - can be improved with NLP libraries
        words = re.findall(r'\b[а-яё]{3,}\b', text.lower())

        # Remove common stop words
        stop_words = {
            'это', 'что', 'как', 'для', 'или', 'при', 'все', 'еще', 'уже',
            'где', 'кто', 'чем', 'том', 'тем', 'так', 'был', 'была', 'было',
            'есть', 'быть', 'мне', 'нас', 'вас', 'них', 'его', 'её', 'их',
            'могу', 'можем', 'можете', 'могут', 'хочу', 'хотим', 'хотите', 'хотят'
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # Return unique keywords, sorted by frequency
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]

    def prepare_profile_text(self, answer_1: str, answer_2: str, answer_3: str) -> Dict[str, Any]:
        """
        Prepare profile text for embedding creation

        Returns:
            Dict with processed text, chunks, and metadata
        """
        # Clean individual answers
        clean_answer_1 = self.clean_text(answer_1)
        clean_answer_2 = self.clean_text(answer_2)
        clean_answer_3 = self.clean_text(answer_3)

        # Create structured text for embedding
        structured_text = f"""Сфера деятельности: {clean_answer_1}
Что ищет в сообществе: {clean_answer_2}
Чем может помочь другим: {clean_answer_3}"""

        # Create chunks if text is long (more than 400 characters)
        if len(structured_text) > 400:
            chunks = self.chunk_text(structured_text)
            logger.info(f"📄 CHUNKING | Text split into {len(chunks)} chunks")
        else:
            chunks = [structured_text]

        # Extract keywords from all answers
        all_text = f"{clean_answer_1} {clean_answer_2} {clean_answer_3}"
        keywords = self.extract_keywords(all_text)

        return {
            'structured_text': structured_text,
            'chunks': chunks,
            'keywords': keywords,
            'total_length': len(structured_text),
            'clean_answers': {
                'answer_1': clean_answer_1,
                'answer_2': clean_answer_2,
                'answer_3': clean_answer_3
            }
        }

    def create_search_query(self, user_profile: Dict[str, str]) -> str:
        """Create optimized search query from user profile"""
        answer_1 = user_profile.get('answer_1', '')
        answer_2 = user_profile.get('answer_2', '')
        answer_3 = user_profile.get('answer_3', '')

        # Extract key terms for search
        keywords_1 = self.extract_keywords(answer_1)[:3]
        keywords_2 = self.extract_keywords(answer_2)[:3]
        keywords_3 = self.extract_keywords(answer_3)[:3]

        # Create search query emphasizing what user is looking for
        search_parts = []

        if keywords_2:  # What they're looking for is most important
            search_parts.append(f"Ищет: {' '.join(keywords_2)}")

        if keywords_1:  # Their field of activity
            search_parts.append(f"Сфера: {' '.join(keywords_1)}")

        if keywords_3:  # What they can offer
            search_parts.append(f"Предлагает: {' '.join(keywords_3)}")

        return ". ".join(search_parts)


# Global text processor instance
text_processor = TextProcessor()
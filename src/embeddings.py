from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from loguru import logger
from config import LLM_CONFIG
from src.text_processing import text_processor


class EmbeddingService:
    def __init__(self, model_name: str = None):
        """Initialize embedding service with specified model"""
        self.model_name = model_name or LLM_CONFIG["embedding_model"]
        self.model = None

    def load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def create_profile_embedding(self, answer_1: str, answer_2: str, answer_3: str) -> List[float]:
        """Create embedding from user profile answers"""
        if not self.model:
            self.load_model()

        # Process text using text processor
        processed_data = text_processor.prepare_profile_text(answer_1, answer_2, answer_3)

        chunks = processed_data['chunks']

        try:
            if len(chunks) == 1:
                # Single chunk - use as before
                embedding = self.model.encode(chunks[0], normalize_embeddings=True)
                logger.debug(f"🧠 SINGLE EMBEDDING | Length: {processed_data['total_length']} chars")
                return embedding.tolist()
            else:
                # Multiple chunks - create embeddings and average them
                chunk_embeddings = []
                for i, chunk in enumerate(chunks):
                    chunk_embedding = self.model.encode(chunk, normalize_embeddings=True)
                    chunk_embeddings.append(chunk_embedding)
                    logger.debug(f"🧠 CHUNK {i + 1}/{len(chunks)} | Length: {len(chunk)} chars")

                # Average the embeddings
                import numpy as np
                averaged_embedding = np.mean(chunk_embeddings, axis=0)
                # Normalize the result
                averaged_embedding = averaged_embedding / np.linalg.norm(averaged_embedding)

                logger.info(f"🧠 AVERAGED EMBEDDING | {len(chunks)} chunks → 1 vector")
                return averaged_embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise

    def create_chunked_embeddings(self, text: str) -> List[List[float]]:
        """Create embeddings for text chunks"""
        if not self.model:
            self.load_model()

        chunks = text_processor.chunk_text(text)
        embeddings = []

        for chunk in chunks:
            try:
                embedding = self.model.encode(chunk, normalize_embeddings=True)
                embeddings.append(embedding.tolist())
            except Exception as e:
                logger.error(f"Failed to create chunk embedding: {e}")
                continue

        return embeddings

    def create_search_embedding(self, user_profile: dict) -> List[float]:
        """Create optimized embedding for search queries"""
        if not self.model:
            self.load_model()

        # Create search query from profile
        search_query = text_processor.create_search_query(user_profile)

        try:
            embedding = self.model.encode(search_query, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create search embedding: {e}")
            raise

    def create_text_embedding(self, text: str) -> List[float]:
        """Create embedding from any text"""
        if not self.model:
            self.load_model()

        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create text embedding: {e}")
            raise


# Global embedding service instance
embedding_service = EmbeddingService()
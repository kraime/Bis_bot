from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
from loguru import logger
from config import QDRANT_URL, QDRANT_API_KEY, DB_CONFIG


class VectorDatabase:
    def __init__(self):
        """Initialize Qdrant client"""
        # Use local Qdrant by default
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY  # None for local instance
        )
        self.collection_name = DB_CONFIG["collection_name"]
        self.vector_size = DB_CONFIG["vector_dimension"]

    async def initialize(self):
        """Initialize Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    def save_profile_embedding(self, user_id: int, embedding: List[float],
                               profile_data: Dict[str, Any]):
        """Save user profile embedding to Qdrant"""
        try:
            # Check if point already exists
            try:
                existing_point = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[user_id]
                )
                if existing_point:
                    logger.info(f"Updating existing embedding for user {user_id}")
                else:
                    logger.info(f"Creating new embedding for user {user_id}")
            except Exception:
                logger.info(f"Creating new embedding for user {user_id}")

            point = PointStruct(
                id=user_id,  # Use user_id as unsigned integer
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "telegram_id": profile_data.get("telegram_id"),
                    "username": profile_data.get("username"),
                    "first_name": profile_data.get("first_name"),
                    "last_name": profile_data.get("last_name"),
                    "answer_1": profile_data.get("answer_1"),
                    "answer_2": profile_data.get("answer_2"),
                    "answer_3": profile_data.get("answer_3"),
                    "keywords": profile_data.get("keywords", [])
                }
            )

            # Use upsert to insert or update
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Saved/updated embedding for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to save embedding for user {user_id}: {e}")
            raise

    def search_similar_profiles(self, embedding: List[float], user_id: int,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar profiles using vector similarity"""
        if embedding is None:
            logger.error(f"Cannot search with None embedding for user {user_id}")
            return []

        try:
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit + 1,  # +1 because we'll filter out the user's own profile
                with_payload=True,
                with_vectors=False
            )

            # Filter out user's own profile and convert to dict
            similar_profiles = []
            for hit in search_result:
                if hit.payload.get("user_id") != user_id:
                    profile = dict(hit.payload)
                    profile["similarity_score"] = hit.score
                    similar_profiles.append(profile)

            return similar_profiles[:limit]

        except Exception as e:
            logger.error(f"Failed to search similar profiles: {e}")
            return []

    def delete_profile(self, user_id: int):
        """Delete user profile from Qdrant"""
        try:
            # Check if profile exists first
            try:
                existing_point = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[user_id]
                )
                if not existing_point:
                    logger.warning(f"Profile {user_id} not found in Qdrant")
                    return False
            except Exception:
                logger.warning(f"Profile {user_id} not found in Qdrant")
                return False

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[user_id]
                )
            )
            logger.info(f"🗑️ QDRANT DELETE | Profile {user_id} deleted from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Failed to delete profile {user_id} from Qdrant: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def get_user_embedding(self, user_id: int) -> Optional[List[float]]:
        """Get user's embedding vector from Qdrant"""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[user_id],
                with_vectors=True
            )

            if points and len(points) > 0:
                return points[0].vector
            else:
                logger.warning(f"No embedding found for user {user_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get embedding for user {user_id}: {e}")
            return None


# Global vector database instance
vector_db = VectorDatabase()
#!/usr/bin/env python3
"""
Setup script for Qdrant vector database
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.database import db
from src.vector_db import vector_db


async def setup_databases():
    """Setup SQLite and Qdrant databases"""
    print("🚀 Setting up SQLite + Qdrant databases...\n")

    try:
        # Setup SQLite
        print("🔍 Setting up SQLite database...")
        await db.connect()
        print("✅ SQLite database initialized")

        # Setup Qdrant
        print("\n🔍 Setting up Qdrant vector database...")
        await vector_db.initialize()
        print("✅ Qdrant collection initialized")

        # Get collection info
        info = vector_db.get_collection_info()
        if info:
            print(f"📊 Collection info: {info}")

        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_bot_simple.py")
        print("2. Run: python main.py")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
    finally:
        await db.close()


def check_qdrant_connection():
    """Check if Qdrant is running"""
    print("🔍 Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print("✅ Qdrant is running")
        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("\nTo start Qdrant locally:")
        print("1. Download Qdrant from: https://github.com/qdrant/qdrant/releases")
        print("2. Extract and run: qdrant")
        print("3. Or run: python install_qdrant.py for installation help")
        return False


if __name__ == "__main__":
    # Check Qdrant first
    if not check_qdrant_connection():
        sys.exit(1)

    # Setup databases
    asyncio.run(setup_databases())
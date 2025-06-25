# Create test_db.py
import asyncio
from database_manager import DatabaseManager
from dotenv import load_dotenv

async def test():
    load_dotenv()
    db = DatabaseManager()
    success = await db.initialize()
    if success:
        print("✅ Database connection successful!")
        connection_ok = await db.test_connection()
        print(f"Connection test: {'✅ OK' if connection_ok else '❌ Failed'}")
        await db.close()
    else:
        print("❌ Database connection failed!")

if __name__ == "__main__":
    asyncio.run(test())
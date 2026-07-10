"""
NeuroAssist AI v2 — MongoDB async connection via Motor.
Falls back to a local JSON file-based mock database if MongoDB is not running.
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from backend.config import settings
from loguru import logger

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None
_use_fallback: bool = False


# =====================================================================
# Mock Database Fallback (for local demo without running MongoDB)
# =====================================================================

class MockCursor:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.index = 0

    def sort(self, key_or_list, direction=None):
        if isinstance(key_or_list, str):
            reverse = (direction == -1)
            self.data.sort(key=lambda x: x.get(key_or_list, ""), reverse=reverse)
        elif isinstance(key_or_list, list) and key_or_list:
            key, dir_val = key_or_list[0]
            reverse = (dir_val == -1)
            self.data.sort(key=lambda x: x.get(key, ""), reverse=reverse)
        return self

    def limit(self, limit_num: int):
        self.data = self.data[:limit_num]
        return self

    def skip(self, skip_num: int):
        self.data = self.data[skip_num:]
        return self

    async def to_list(self, length: int = None):
        if length is not None:
            return self.data[:length]
        return self.data

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index < len(self.data):
            val = self.data[self.index]
            self.index += 1
            return val
        raise StopAsyncIteration


class MockCollection:
    def __init__(self, name: str):
        self.name = name
        self.file_path = f"./db_mock/{name}.json"
        os.makedirs("./db_mock", exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_data(self, data: List[Dict[str, Any]]):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _match_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for k, v in query.items():
            if k == "_id" and doc.get("_id") != v:
                return False
            elif isinstance(v, dict):
                if "$in" in v:
                    if doc.get(k) not in v["$in"]:
                        return False
            elif doc.get(k) != v:
                return False
        return True

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await asyncio.sleep(0.005)
        data = self._read_data()
        for doc in data:
            if self._match_query(doc, query):
                return doc
        return None

    async def insert_one(self, document: Dict[str, Any]):
        await asyncio.sleep(0.005)
        data = self._read_data()
        data.append(document)
        self._write_data(data)
        
        class InsertOneResult:
            inserted_id = document.get("_id")
        return InsertOneResult()

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        await asyncio.sleep(0.005)
        data = self._read_data()
        found = False
        for doc in data:
            if self._match_query(doc, query):
                found = True
                if "$set" in update:
                    for k, v in update["$set"].items():
                        doc[k] = v
                break
        if not found and upsert:
            new_doc = query.copy()
            if "$set" in update:
                for k, v in update["$set"].items():
                    new_doc[k] = v
            data.append(new_doc)
        self._write_data(data)
        
        class UpdateResult:
            modified_count = 1 if found else 0
            matched_count = 1 if found else 0
        return UpdateResult()

    async def delete_one(self, query: Dict[str, Any]):
        await asyncio.sleep(0.005)
        data = self._read_data()
        new_data = []
        deleted = False
        for doc in data:
            if self._match_query(doc, query) and not deleted:
                deleted = True
                continue
            new_data.append(doc)
        self._write_data(new_data)
        
        class DeleteResult:
            deleted_count = 1 if deleted else 0
        return DeleteResult()

    def find(self, query: Dict[str, Any] = None, projection: Dict[str, Any] = None, *args, **kwargs):
        if query is None:
            query = {}
        data = self._read_data()
        matched = [doc for doc in data if self._match_query(doc, query)]
        if projection:
            # Handle standard field exclusions (e.g. {"password_hash": 0})
            for doc in matched:
                for k, v in list(projection.items()):
                    if v == 0:
                        doc.pop(k, None)
        return MockCursor(matched)

    async def count_documents(self, query: Dict[str, Any]):
        await asyncio.sleep(0.005)
        data = self._read_data()
        matched = [doc for doc in data if self._match_query(doc, query)]
        return len(matched)


# =====================================================================
# Database Actions
# =====================================================================

async def connect_to_mongo() -> None:
    """Initialize the MongoDB connection."""
    global _client, _database, _use_fallback
    logger.info(f"Connecting to MongoDB at {settings.mongodb_uri[:30]}...")
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _database = _client[settings.mongodb_db_name]

    # Verify connection
    try:
        # 2-second timeout for quick fallback detection
        await asyncio.wait_for(_client.admin.command("ping"), timeout=2.0)
        logger.info("MongoDB connection established successfully.")
        _use_fallback = False
    except Exception as e:
        logger.warning(
            f"MongoDB connection failed: {e}. "
            "Falling back to local JSON database mock at ./db_mock/"
        )
        _use_fallback = True


async def close_mongo_connection() -> None:
    """Close the MongoDB connection."""
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance. Raises if not connected."""
    if _database is None:
        raise RuntimeError(
            "MongoDB not connected. Call connect_to_mongo() first."
        )
    return _database


# ---- Collection accessors ----

def get_doctors_collection():
    if _use_fallback:
        return MockCollection("doctors")
    return get_database()["doctors"]


def get_patients_collection():
    if _use_fallback:
        return MockCollection("patients")
    return get_database()["patients"]


def get_predictions_collection():
    if _use_fallback:
        return MockCollection("predictions")
    return get_database()["predictions"]


def get_reports_collection():
    if _use_fallback:
        return MockCollection("reports")
    return get_database()["reports"]


def get_risk_assessments_collection():
    if _use_fallback:
        return MockCollection("risk_assessments")
    return get_database()["risk_assessments"]


def get_chat_sessions_collection():
    if _use_fallback:
        return MockCollection("chat_sessions")
    return get_database()["chat_sessions"]


def get_audit_log_collection():
    if _use_fallback:
        return MockCollection("audit_log")
    return get_database()["audit_log"]


def get_model_metadata_collection():
    if _use_fallback:
        return MockCollection("model_metadata")
    return get_database()["model_metadata"]

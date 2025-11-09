from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
import logging

class DatabaseManager:
    def __init__(self, uri: str, db_name: str, collection_name: str):

        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self):
        try:
            self.client = MongoClient(self.uri)
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logging.info(f"Connected to MongoDB: {self.db_name}.{self.collection_name}")
        except ConnectionFailure as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise e

    def insert_result(self, result_doc: dict):
        try:
            # Add timestamp if not present
            if 'timestamp' not in result_doc:
                result_doc['timestamp'] = datetime.now().isoformat()
            if 'processed_at' not in result_doc:
                result_doc['processed_at'] = datetime.now().isoformat()

            result = self.collection.insert_one(result_doc)
            logging.info(f"Inserted document with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logging.error(f"Failed to insert document: {e}")
            raise e

    def find_results(self, query: dict = None, limit: int = 10):
        if query is None:
            query = {}
        try:
            results = list(self.collection.find(query).limit(limit))
            return results
        except Exception as e:
            logging.error(f"Failed to query documents: {e}")
            raise e

    def close(self):
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed")

def create_database_manager(uri: str, db_name: str, collection_name: str) -> DatabaseManager:
    return DatabaseManager(uri, db_name, collection_name)

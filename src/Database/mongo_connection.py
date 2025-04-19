from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()


class MongoConnection:
    _instance = None

    @staticmethod
    def get_instance():
        if MongoConnection._instance is None:
            MongoConnection()
        return MongoConnection._instance

    def __init__(self):
        if MongoConnection._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            mongo_uri = os.getenv("MONGO_URI")
            db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            MongoConnection._instance = self

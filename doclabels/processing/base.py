import abc
import csv
import json
import logging
import numpy as np
import settings
import sys
import time
import yaml
from celery import Celery
from time import strftime
from pymongo import MongoClient


logging.basicConfig(filename='log/process.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BaseDataBaseManager(object):
    """
    Manage Mongo db.
    """


class BaseProcessor(object):
    """
    Process raw and preprocessed documents.
    """

    def save_raw(self):
        """
        Save raw data (to MongoDB)
        """
    def save_preprocessed(self):
        """
        Save preprocessed
        """


class BaseHarvester(object):
    """
    Base Processor with abstract methods.
    """
    @abc.abstractmethod
    def harvest(self):
        """
        Download source data
        """

    @abc.abstractmethod
    def process(self):
        """
        Process raw data.
        """


class MongoManager(BaseDataBaseManager):
    def __init__(self, uri=None, timeout=None, database=None, **kwargs):
        self.uri = settings.MONGO_URI
        self.database = settings.MONGO_DATABASE
        self.client = None
        self.db = None
        self.collection = None
        self.collection_raw = None

    def setup(self):
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.database]
            try:
                db.doclabels.getIndexes()
            except:
                self.db.doclabels.create_index('doc_id', unique=True)
            try:
                db.doclabels_raw.getIndexes()
            except:
                self.db.doclabels_raw.create_index('doc_id', unique=True)
            try:
                self.db.counters.insert({'_id': 'item_id', 'seq': 0})
            except:
                pass
        except:
            logging.error("Failed to connect to Mongodb.")
            raise

    def tear_down(self, force=False):
        pass

    def clear(self, force=False):
        assert force, "Force must be called."
        self.client.drop_database(self.db)

    def celery_setup(self, *args, **kwargs):
        pass

    def get_next_sequence(self, name):
        return self.db.counters.find_and_modify(
            query={'_id': name},
            update={'$inc': {'seq': 1}},
            new=True)['seq']


class MongoProcessor(BaseProcessor):
    manager = MongoManager()

    def save_raw(self, raw, item_id):
        labels = raw.pop('labels', None)
        stamp = raw.pop('stamp', None)
        raw['_id'] = item_id
        self.manager.db.doclabels_raw.update_one(
            {'doc_id': raw['doc_id']},
            {'$setOnInsert': raw, '$addToSet': {'labels': {'$each': labels}}, '$push': {'stamp': {'$each': stamp}}},
            upsert=True)

    def save_preprocessed(self, processed, item_id):
        labels = processed.pop('labels', None)
        stamp = processed.pop('stamp', None)
        processed['_id'] = item_id
        self.manager.db.doclabels.update_one(
            {'doc_id': processed['doc_id']},
            {'$setOnInsert': processed, '$addToSet': {'labels': {'$each': labels}}, '$push': {'stamp': {'$each': stamp}}},
            upsert=True)

    def save(self, raw, processed):
        item_id = self.manager.get_next_sequence("item_id")
        self.save_raw(raw, item_id)
        self.save_preprocessed(processed, item_id)

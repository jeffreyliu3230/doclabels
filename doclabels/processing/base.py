import abc
import csv
import json
import logging
import settings
import sys
import time
import yaml
from celery import Celery
from time import strftime
from pymongo import MongoClient
from random import SystemRandom

logging.basicConfig(filename='log/process.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

random = SystemRandom()


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

    def setup(self, client=None):
        try:
            self.client = client or MongoClient(self.uri)
            self.db = self.client[self.database]
            self.collection = self.db[settings.MONGO_COLLECTION]
            self.collection_raw = self.db[settings.MONGO_COLLECTION_RAW]
            try:
                self.collection.getIndexes()
            except:
                self.collection.create_index('doc_id', unique=True)
            try:
                self.collection_raw.getIndexes()
            except:
                self.collection_raw.create_index('doc_id', unique=True)
            try:
                self.db.counters.insert({'_id': 'item_id', 'seq': 0})
            except:
                pass
        except:
            logging.error("Failed to connect to Mongodb.")
            raise

    def close(self):
        """
        Close connection.
        """
        self.client.close()

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
        self.manager.collection_raw.update_one(
            {'doc_id': raw['doc_id']},
            {'$setOnInsert': raw, '$addToSet': {'labels': {'$each': labels}}, '$push': {'stamp': {'$each': stamp}}},
            upsert=True)

    def save_preprocessed(self, processed, item_id):
        labels = processed.pop('labels', None)
        stamp = processed.pop('stamp', None)
        processed['_id'] = item_id
        self.manager.collection.update_one(
            {'doc_id': processed['doc_id']},
            {'$setOnInsert': processed, '$addToSet': {'labels': {'$each': labels}}, '$push': {'stamp': {'$each': stamp}}},
            upsert=True)

    def save(self, raw, processed):
        item_id = self.manager.get_next_sequence("item_id")
        self.save_raw(raw, item_id)
        self.save_preprocessed(processed, item_id)

    def fetch_ids(self):
        return [i['_id'] for i in self.manager.collection.find({}, {'_id': 1})]

    def stream_documents(self, ids=None):
        """
        Stream documents from db given a list of ids in random order.
        """
        ids = ids or self.fetch_ids()
        for objectid in ids:
            yield self.manager.collection.find_one({'_id': objectid})

    def iter_batch(self, ids, batch_size=settings.DEFAULT_BATCH_SIZE, epochs=settings.DEFAULT_EPOCHS):
        """
        Yield data one batch at a time.
        """
        data_size = len(ids)
        num_batches_per_epoch = int(data_size / batch_size) + 1
        for epoch in range(epochs):
            # Shuffle the indices at each epoch
            shuffled_indices = random.sample(ids, data_size)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if not start_index == end_index:
                    yield list(self.manager.collection.find({'_id': {'$in': shuffled_indices[start_index:end_index]}}))

    def batch_xy(self, ids, batch_size=settings.DEFAULT_BATCH_SIZE, epochs=settings.DEFAULT_EPOCHS, use_title=True):
        """
        Yield explanatory and response data separately.
        """
        data_size = len(ids)
        num_batches_per_epoch = int(data_size / batch_size) + 1
        for epoch in range(epochs):
            # Shuffle the indices at each epoch
            shuffled_indices = random.sample(ids, data_size)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if not start_index == end_index:
                    docs = self.manager.collection.find({'_id': {'$in': shuffled_indices[start_index:end_index]}})
                    y = [doc['labels'] for doc in docs]
                    docs.rewind()
                    if use_title:
                        x = [doc['title'] + doc['doc'] for doc in docs]
                    else:
                        x = [doc['doc'] for doc in docs]
                    docs.close()
                    yield x, y

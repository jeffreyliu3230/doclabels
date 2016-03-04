import abc
import csv
import json
import logging
import numpy as np
import settings
import sys
import time
import yaml
from elasticsearch import Elasticsearch, helpers
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
                self.db.doclabels.create_index('id', unique=True)
            try:
                db.doclabels_raw.getIndexes()
            except:
                self.db.doclabels_raw.create_index('id', unique=True)
            try:
                self.db.counters.insert({'_id': 'docid', 'seq': 0})
            except:
                pass
        except:
            logging.error("Failed to connect to Mongodb.")
            raise

    def tear_down(self, dbname=None):
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
            new=True).seq


class ElasticsearchManager(BaseDataBaseManager):
    def __init__(self, uri=None, timeout=None, index=None, raw=None, **kwargs):
        self.uri = uri or settings.ELASTIC_URI
        self.index = index or settings.ELASTIC_INDEX
        self.raw = raw or settings.ELASTIC_INDEX_RAW
        self.es = None
        self.kwargs = {
            'timeout': timeout or settings.ELASTIC_TIMEOUT,
            'retry_on_timeout': True
        }
        self.kwargs.update(kwargs)

    def setup(self):
        """
        Sets up the database connection. Returns True if the database connection
        is successful, False otherwise
        """
        try:
            # If we cant connect to elastic search dont define this class
            self.es = Elasticsearch(self.uri, **self.kwargs)

            self.es.cluster.health(wait_for_status='yellow')
            self.es.indices.create(index=self.index, body={}, ignore=400)
            return True
        except ConnectionError:  # pragma: no cover
            logger.error('Could not connect to Elasticsearch, expect errors.')
            return False

    def tear_down(self):
        """
        since it's just http, doesn't do much
        """
        pass

    def clear(self, index, force=False):
        assert force, 'Force must be called to clear the database'
        # assert self.index != settings.ELASTIC_INDEX, 'Cannot erase the production database'
        self.es.indices.delete(index=index, ignore=[400, 404])

    def celery_setup(self, *args, **kwargs):
        pass


class MongoProcessor(BaseProcessor):
    manager = MongoManager()

    def save_raw(self, raw, docid):
        labels = raw.pop('labels', None)
        stamp = raw.pop('stamp', None)
        self.manager.db.doclabels_raw.update_one(
            {'_id': docid},
            {'$setOnInsert': raw, '$addToSet': {'labels': {'$each': labels}}, '$push': {'stamp': {'$each': stamp}}},
            upsert=True)

    def save_preprocessed(self, processed, docid):
        labels = processed.pop('labels', None)
        stamp = processed.pop('stamp', None)
        self.manager.db.doclabels.update_one(
            {'_id': processed['id']},
            {'$setOnInsert': processed, '$addToSet': {'labels': {'$each': labels}}, '$push': {'stamp': {'$each': stamp}}},
            upsert=True)

    def save(self, raw, processed):
        docid = manager.get_next_sequence("docid")
        self.save_raw(raw)
        self.save_processed(processed)


class ElasticsearchProcessor(BaseProcessor):
    """
    Deprecated. Now data are injected through mongo-connector.
    """
    manager = ElasticsearchManager()
    app = Celery('elastic_search')
    app.config_from_object(settings)

    def save_preprocessed(self, processed, doc_type, index=None):
        index = index or self.manager.index

        self.manager.es.index(
            body=processed,
            refresh=True,
            index=index,
            doc_type=doc_type,
            id=processed['id'])

    def batch_save_prerpocessed(self, doc_iter, doc_type, action='update', index=None):
        for acts in actions_wrapper(action, doc_iter, doc_type):
            async_bulk_wrapper.delay(acts)

    def construct_action(action, doc_type, doc, index):
        return {
            '_op_type': action,
            '_index': self.index,
            '_type': doc_type,
            '_id': doc['id'],
            'doc': doc,
            'upsert': doc
        }

    def actions(action, doc_iter, doc_type):
        for doc in doc_iter:
            yield construct_action(action, doc_type, doc, index=self.index)

    @app.task(bind=True, default_retry_delay=30)
    def async_bulk_wrapper(self, acts):
        try:
            return helpers.bulk(self.manager.es, acts, stats_only=True)
        except Exception as e:
            logger.info(e)
            raise self.retry(exc=e)

    def actions_wrapper(action, doc_iter, doc_type, size=500):
        count = 0
        acts = []
        for action in actions(action, doc_iter, doc_type):
            acts.append(action)
            count += 1
            if count == size:
                yield acts
                count = 0
                acts = []

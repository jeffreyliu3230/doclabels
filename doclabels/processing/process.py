import abc
import csv
import json
import logging
import numpy as np
import settings
import sys
import time
import yaml
from doclabels.processing.base import MongoProcessor, ElasticsearchProcessor
from doclabels.processing.harvesters.plos import PLOSHarvester
from time import strftime
from pymongo import MongoClient

logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(filename='log/process.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

__defaultInc__ = 500
__defaultLimit__ = 500
__defaultStart__ = 0


def process_plos(limit=__defaultInc__, increment=__defaultLimit__, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=__defaultStart__, async=False):
    settings.CELERY_ALWAYS_EAGER = not async
    harvester = PLOSHarvester()
    mongoprocessor = MongoProcessor()
    elasticprocessor = ElasticsearchProcessor()
    mongoprocessor.manager.setup()
    elasticprocessor.manager.setup()
    doc_iter = harvester.harvest(stamp=stamp)
    if async:
        mongoprocessor.batch_save_raw(doc_iter)
        elasticprocessor.batch_save_preprocessed(doc_iter)
    else:
        for doc in doc_iter:
            mongoprocessor.save_raw(doc['raw'])
            elasticprocessor.save_preprocessed(doc['preprocessed'], doc_type=settings.PLOS_DOC_TYPE)


def file_to_db():
    """
    migrate data from files to es and mongo.
    """

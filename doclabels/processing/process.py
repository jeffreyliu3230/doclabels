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

DEFAULT_INC = 500
DEFAULT_LIMIT = 500
DEFAULT_START = 0


def process_plos(limit=DEFAULT_INC, increment=DEFAULT_LIMIT, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=DEFAULT_START, async=False):
    settings.CELERY_ALWAYS_EAGER = not async
    harvester = PLOSHarvester()
    mongoprocessor = MongoProcessor()
    mongoprocessor.manager.setup()
    doc_iter = harvester.harvest(limit=limit, increment=increment, stamp=stamp, subject_areas=subject_areas, start=start)
    if async:
        mongoprocessor.batch_save_raw(doc_iter)
        elasticprocessor.batch_save_preprocessed(doc_iter)
    else:
        for doc in doc_iter:
            mongoprocessor.save_raw(doc['raw'])
            mongoprocessor.save_preprocessed(doc['preprocessed'])


def file_to_db():
    """
    migrate data from files to es and mongo.
    """

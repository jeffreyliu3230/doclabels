import abc
import csv
import json
import logging
import numpy as np
import settings
import sys
import time
import yaml
from doclabels.processing.base import MongoProcessor
from doclabels.processing.harvesters.plos import PLOSHarvester
from doclabels.helpers import generate_response_map, to_classes
from time import strftime

logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(filename='log/process.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_plos(client, limit=settings.DEFAULT_LIMIT, increment=settings.DEFAULT_INC,
                 stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS,
                 start=settings.DEFAULT_START, async=False):
    """
    Save PLOS data to db.
    """
    settings.CELERY_ALWAYS_EAGER = not async
    harvester = PLOSHarvester()
    mongoprocessor = MongoProcessor()
    mongoprocessor.manager.setup(client)
    doc_iter = harvester.harvest(limit=limit, increment=increment, stamp=stamp, subject_areas=subject_areas,
                                 start=start)
    if async:
        pass
    else:
        for doc in doc_iter:
            mongoprocessor.save(doc['raw'], doc['preprocessed'])


def create_classes(client, collection):
    """
    Generate response variable from labels. Text => integers.
    """
    mongoprocessor = MongoProcessor()
    mongoprocessor.manager.setup(client)
    for doc in mongoprocessor.manager.db[collection].find({}):
        labels = doc['labels']
        classes = to_classes(labels, generate_response_map(settings.SUBJECT_AREAS))
        mongoprocessor.manager.db.classes.insert({'_id': doc['_id'], 'classes': classes})

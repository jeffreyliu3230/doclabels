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
from time import strftime

logging.getLogger().addHandler(logging.StreamHandler())
logging.basicConfig(filename='log/process.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_plos(limit=settings.DEFAULT_INC, increment=settings.DEFAULT_LIMIT, stamp=str(strftime("%Y%m%d%H%M%S")),
                 subject_areas=settings.SUBJECT_AREAS, start=settings.DEFAULT_START, async=False):
    settings.CELERY_ALWAYS_EAGER = not async
    harvester = PLOSHarvester()
    mongoprocessor = MongoProcessor()
    mongoprocessor.manager.setup()
    doc_iter = harvester.harvest(limit=limit, increment=increment, stamp=stamp, subject_areas=subject_areas,
                                 start=start)
    if async:
        pass
    else:
        for doc in doc_iter:
            mongoprocessor.save(doc['raw'], doc['preprocessed'])

import csv
import json
import logging
import numpy as np
import settings
import sys
import time
import yaml
from doclabels.processing.base import BaseHarvester
from doclabels.processing.harvesters import plosapi
from doclabels.helpers import compose, clean_str
from time import strftime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PLOSHarvester(BaseHarvester):
    """
    Process data from PLOS API.
    """
    SOURCE = 'plos'
    DEFAULT_INC = settings.DEFAULT_INC
    DEFAULT_LIMIT = settings.DEFAULT_LIMIT
    DEFAULT_START = settings.DEFAULT_START

    def harvest(self, limit=DEFAULT_LIMIT, increment=DEFAULT_INC, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=DEFAULT_START):
        """
        Download data from plos api.
        """
        tick = time.clock()
        if isinstance(subject_areas, str):
            subject_areas = [subject_areas]
        for subject in subject_areas:
            logger.info('{} documents to be saved for the subject: {}'.format(limit, subject))
            for docs in plosapi.sample('subject:\"{}\"'.format(subject), limit, increment, start):
                for doc in docs:
                    # yield doc
                    yield {
                        'raw': {'doc_id': doc['id'], 'doc': doc, 'labels': [subject], 'stamp': [stamp], 'source': self.SOURCE},
                        'preprocessed': self.process(doc, subject, stamp)
                    }
                logger.info("{} results returned in time: {}.".format(limit, time.clock() - tick))
                time.sleep(1)
        logger.info('PLOS data harvested. time: {}\n'.format(time.clock() - tick))
        time.sleep(10)

    def process(self, doc, subject, stamp, pad=False):
        """
        Prepare input data for scikit-learn classifiers.
        """
        return {
            'doc_id': doc['id'],
            'title': compose(lambda x: x.split(" "), clean_str)(doc['title_display']),
            'doc': compose(lambda x: x.split(" "), clean_str)(doc['abstract'][0]),
            'labels': [subject],
            'stamp': [stamp],
            'source': self.SOURCE
        }

    def batch_process(self, docs, subject, pad=False):
        return map(lambda doc: self.process(doc, subject, pad), docs)

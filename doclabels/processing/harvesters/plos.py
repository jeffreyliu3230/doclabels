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
    __defaultInc__ = 500
    __defaultLimit__ = 500
    __defaultStart__ = 0

    def harvest(self, limit=__defaultInc__, increment=__defaultLimit__, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=__defaultStart__):
        """
        Download data from plos api.
        """
        tick = time.clock()
        if isinstance(subject_areas, str):
            subject_areas = [subject_areas]
        for subject in subject_areas:
            logger.info('{} documents to be saved for the subject: {}'.format(limit, subject))
            subject_collection = subject.replace(' ', '_').lower()
            for docs in plosapi.sample('subject:\"{}\"'.format(subject), limit, increment, start):
                for doc in docs:
                    # yield doc
                    yield {
                        'raw': {'id': doc['id'], 'doc': doc, 'subject': subject, 'stamp': stamp},
                        'preprocessed': self.process(doc, subject, stamp)
                    }
                logger.info("{} results returned in time: {}.".format(limit, time.clock() - tick))
                time.sleep(1)
        logger.info('PLOS data harvested. time: {}\n'.format(time.clock() - tick))

    def process(self, doc, subject, stamp, pad=False):
        """
        Prepare input data for scikit-learn classifiers.
        """
        return {
            'id': doc['id'],
            'title': compose(lambda x: x.split(" "), clean_str)(doc['title_display']),
            'doc': compose(lambda x: x.split(" "), clean_str)(doc['abstract'][0]),
            'subject': [subject],
            'stamp': stamp
        }

    def batch_process(self, docs, subject, pad=False):
        return map(lambda doc: self.process(doc, subject, pad), docs)

import logging
import time
import settings

from invoke import run, task
from time import strftime


@task
def start_services():
    """
    Start MongoDB, Elasticsearch.
    """


@task
def drop_mongo():
    """
    Drop doclabels.
    """
    from doclabels.processing.base import MongoManager
    mm = MongoManager()
    mm.setup()
    mm.clear(force=True)


@task
def process_plos(limit=settings.DEFAULT_LIMIT, increment=settings.DEFAULT_INC, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=settings.DEFAULT_START, async=False):
    """
    Save PLOS data to MongoDB.
    """
    from doclabels.processing.process import process_plos
    process_plos(limit=limit, increment=increment, stamp=stamp, subject_areas=subject_areas, start=start, async=async)


@task
def mass_process_plos(limit=settings.DEFAULT_LIMIT, increment=settings.DEFAULT_INC, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=settings.DEFAULT_START, async=False):
    """
    Use task-spooler, split the harvester job into small chunks.
    """
    import os
    for i in xrange(start, start + limit, increment):
        os.system('ts -L {}:{}-{}/{} inv process_plos --limit {} --increment {} --stamp {} --start {}'.format(
            start, i - start, i - start + increment, limit, increment, increment, stamp, i))
    os.system('ts -l')


@task
def get_num(subject_areas=settings.SUBJECT_AREAS):
    """
    Get number of documents in plos database.
    """
    from doclabels.processing.harvesters.plosapi import get_num
    total = 0
    for subject in subject_areas:
        doc_num = get_num('subject:\"{}\"'.format(subject))
        total += doc_num
        print("subject: {}, number of docs: {}".format(subject, doc_num))
    print("total: {}".format(total))

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
    from doclabels.processing.base import MongoManager
    mm = MongoManager()
    mm.setup()
    mm.clear(force=True)


@task
def process_plos(limit=settings.DEFAULT_INC, increment=settings.DEFAULT_LIMIT, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=settings.DEFAULT_START, async=False):
    from doclabels.processing.process import process_plos
    process_plos(limit=limit, increment=increment, stamp=stamp, subject_areas=subject_areas, start=start, async=async)


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

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
def drop_all():
    """
    Drop all databases. For test use.
    """
    from doclabels.processing.base import ElasticsearchManager, MongoManager
    em = ElasticsearchManager()
    mm = MongoManager()
    em.setup()
    mm.setup()
    em.clear(settings.MONGO_DATABASE, force=True)
    mm.clear(settings.ELASTIC_INDEX, force=True)
    em.clear(settings.MONGO_DATABASE_RAW, force=True)
    mm.clear(settings.ELASTIC_INDEX_RAW, force=True)


@tast
def drop_mongo():
    from doclabels.processing.base import MongoManager
    mm = MongoManager()
    mm.setup()
    em.clear(settings.MONGO_DATABASE, force=True)
    em.clear(settings.MONGO_DATABASE_RAW, force=True)


@task
def process_plos(limit=settings.DEFAULT_INC, increment=settings.DEFAULT_LIMIT, stamp=str(strftime("%Y%m%d%H%M%S")), subject_areas=settings.SUBJECT_AREAS, start=settings.DEFAULT_START, async=False):
    from doclabels.processing.process import process_plos
    process_plos(limit=limit, increment=increment, stamp=stamp, subject_areas=subject_areas, start=start, async=async)

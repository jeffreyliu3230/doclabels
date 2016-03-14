import logging
from settings.defaults import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from settings.local import *
except ImportError as error:
    logger.warn("No local settings found. Try create local.py under settings folder.")

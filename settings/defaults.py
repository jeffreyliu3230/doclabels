# Default settings
SUBJECT_AREAS = ('Biology and life sciences', 'Computer and information sciences', 'Earth sciences',
                 'Ecology and environmental sciences', 'Engineering and technology',
                 'Medicine and health sciences', 'People and places', 'Physical sciences',
                 'Research and analysis methods', 'Science policy', 'Social sciences')

# Celery settings
BROKER_URL = 'amqp://localhost'
CELERY_ALWAYS_EAGER = True

# MongoDB
MONGO_URI = 'mongodb://localhost:27017/'
MONGO_DATABASE = 'doclabels'
MONGO_COLLECTION_RAW = 'doclabels_raw'
MONGO_COLLECTION = 'doclabels'

# Elasticsearch
ELASTIC_URI = 'localhost:9200'
ELASTIC_INDEX = 'doclabels'
ELASTIC_INDEX_RAW = 'doclabels_raw'
ELASTIC_TIMEOUT = 10
PLOS_DOC_TYPE = 'plos'

# Harvester
DEFAULT_INC = 500
DEFAULT_LIMIT = 500
DEFAULT_START = 0


# Training
DEFAULT_SEED = 39194861  # For shuffling indices for each epoch
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_BATCH_SIZE = 100
DEFAULT_EPOCHS = 5
DEFAULT_TEST_SIZE = 0.1
DEFAULT_CHECKPOINT = 500
DEFAULT_NGRAM_RANGE = (1, 1)

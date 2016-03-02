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
MONGO_DATABASE_RAW = 'doclabels_raw'

# Elasticsearch
ELASTIC_URI = 'localhost:9200'
ELASTIC_INDEX = 'doclabels'
ELASTIC_TIMEOUT = 10
PLOS_DOC_TYPE = 'plos'

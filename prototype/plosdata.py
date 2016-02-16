# Preprocess data for injection
import json
import logging
import numpy as np
import re
import time

from collections import Counter
from helpers import compose
from itertools import chain
from prototype import plosapi
from urllib2 import quote

reload(plosapi)

__defaultInc__ = 500
__defaultLimit__ = 500
__vecDimension__ = 300

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

subject_areas = ('Biology and life sciences', 'Computer and information sciences', 'Earth sciences',
                 'Ecology and environmental sciences', 'Engineering and technology',
                 'Medicine and health sciences', 'People and places', 'Physical sciences',
                 'Research and analysis methods', 'Science policy', 'Social sciences')


def _generate_response_map(subject_areas):
    return {subject: [0] * i + [1] + [0] * (len(subject_areas) - 1 - i) for i, subject in enumerate(subject_areas)}

response_map = _generate_response_map(subject_areas)


def save_source(filepath, prefix='', subject_areas=subject_areas, limit=__defaultLimit__, increment=__defaultInc__):
    """
    Save source data from PLOS given a subject area of a list of subject areas.
    Can specify the size of the sample you want.
    """
    if isinstance(subject_areas, str):
        subject_areas = [subject_areas]
    for subject in subject_areas:
        logger.info('{} documents to be saved for the subject: {}'.format(limit, subject))
        start = time.clock()
        docs = plosapi.sample('subject:\"{}\"'.format(subject), limit, increment)
        source = {'response': response_map[subject], 'docs': docs}
        filename = subject.replace(" ", "_").lower()
        with open('{}/{}-{}.sample'.format(filepath, prefix, filename), 'w') as f:
            json.dump(source, f)
        logger.info("{} results returned in time: {}.".format(limit, time.clock() - start))


def load_source(filename):
    """
    Load source data from json dump.
    Return filename and data.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def add_response(filename, data):
    """
    add response (subject area) to the data
    """
    return {'response': response_map, 'docs': data}


def get_text(doc):
    return doc['abstract'][0]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("  ", " ", string)
    return string.strip().lower()


# split data in ten folds
def split_data(folds=10, cv=False):
    """
    If cv = True then create training and test sets for cross validation.
    """
    pass


def pad_docs(docs, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in docs)
    return [doc + [padding_word] * (sequence_length - len(doc)) for doc in docs]


def build_vocabulary(docs):
    """
    Build index for each word in the sample data.
    """
    # import ipdb
    # ipdb.set_trace()
    return {x[0]: i for i, x in enumerate(Counter(list(chain.from_iterable(docs))).most_common())}


def build_word_vec_space(vocab, model):
    """
    Build a word vector space for the sample data using the vocabulary and pretrained word vectors (Google News)
    The returned object can be used as the model for the function build_word_vec_index.
    """
    word_vec_space = {}
    for word in vocab:
        if word in model:
            word_vec_space[word] = model[word]
        else:
            word_vec_space[word] = np.random.uniform(-0.25, 0.25, __vecDimension__)
    return word_vec_space


def build_word_vec_index(vocab, model):
    """
    Build a word vector index for the sample data using the vocabulary and pretrained word vectors (Google News)
    """
    word_vec_index = {}
    for word in vocab:
        if word in model:
            word_vec_index[vocab[word]] = model[word]
        else:
            word_vec_index[vocab[word]] = np.random.uniform(-0.25, 0.25, __vecDimension__)
    return word_vec_index


def process_source(sourcefile):
    """
    Process a single source file.
    Return cleaned text (list) and response (list). Split the text.
    """
    source = load_source(sourcefile)
    print('loaded source')
    return map(compose(lambda x: x.split(" "), clean_str, get_text), source['docs']), [source['response'] for i in xrange(len(source['docs']))]


def process_sources(file_list):
    """
    Combine all preprocessing steps here.
    cleaning, add response, padding.
    """
    docs = []
    responses = []
    for sourcefile in file_list:
        print('processing: {}'.format(sourcefile))
        text, response = process_source(sourcefile)
        docs.extend(text)
        responses.extend(response)
    padded = pad_docs(docs)
    vocab = build_vocabulary(padded)

    return padded, responses, vocab


def load_processed(text_file, response_file):
    with open(text_file, 'r') as f1:
        processed = json.load(f1)
    with open(response_file, 'r') as f2:
        responses = json.load(f2)
    return processed, responses


def build_word_matrix(vocab, padded):
    """
    Transform documents to matrices of word indices, which is used as input data for the CNNs.
    """
    return np.array([[vocab[word] for word in doc] for doc in padded])


def build_input(vocab, docs, responses):
    """
    Build x and y of the input data
    """
    return build_word_matrix(vocab, docs), np.array(responses)


def create_embeddings(word_vec_ind):
    """
    Get rid of the keys in word_vec_ind and return a numpy array with all the word vecs in the vocabulary.
    """
    return np.array([word_vec_ind[key] for key in word_vec_ind.keys()])


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset (By Denny Britz).
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

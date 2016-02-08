# Preprocess data for injection
import json
import logging

from helpers import compose
from prototype import plosapi
from urllib2 import quote

__defaultInc__ = 500
__defaultLimit__ = 500
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
    '''
    Save source data from PLOS given a subject area of a list of subject areas.
    Can specify the size of the sample you want.
    '''
    if isinstance(subject_areas, str):
        subject_areas = [subject_areas]
    for subject in subject_areas:
        logger.info('{} documents to be saved for the subject: {}'.format(limit, subject))
        docs = plosapi.sample(subject, limit, increment)
        source = {'response': response_map[subject], 'docs': docs}
        filename = subject.replace(" ", "_").lower()
        with open('{}/{}-{}.sample'.format(filepath, prefix, filename), 'w') as f:
            json.dump(source, f)


def load_source(filename):
    '''
    Load source data from json dump.
    Return filename and data.
    '''
    with open(filename, 'r') as f:
        return json.load(f)


def add_response(filename, data):
    '''
    add response (subject area) to the data
    '''
    return {'response': response_map, 'docs': data}


def get_text(doc):
    return doc['abstract']


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# split data in ten folds
def split_data(folds=10, cv=False):
    '''
    If cv = True then create training and test sets for cross validation.
    '''
    pass


def pad_docs(docs, padding_word="<PAD/>"):
    sequence_length = max(len(x['docs']) for x in docs)
    return [doc + [padding_word] * (squence_length - len(docs)) for i, doc in enumerate(docs)]


def load_word_vecs():
    pass


def process_source(sourcefile):
    '''
    Process a single source file.
    Return cleaned text (list) and response (list).
    '''
    source = load_source(sourcefile)
    return map(compose(clean_str, get_text), source['docs']), source['response'] * len(source['docs'])


def process_sources(file_list):
    '''
    Combine all preprocessing steps here.
    cleaning, add response, word2vec, form matrices, padding,
    '''
    docs = []
    responses = []
    for sourcefile in file_list:
        text, response = process_source(sourcefile)
        docs.extend(text)
        responses.extend(response)
    padded = pad_docs(docs)
    return docs


def save_processed(processed, root='./', prefix=''):
    '''
    save the normalized data (word matrix).
    '''
    json.dump(processed, "")


def build_and_save():
    '''
    create normalized data and save to disk
    '''
    normalized = normalize(data)
    save_normalized(normalized)


def load_normalized(fp, rebuild=False):
    '''
    load normalized data
    '''
    try:
        print("loading data from: {}".format(filename))
        return json.load("{}".format(fp))
    except:
        if rebuild is True:
            normalized = normalize()
            save_normalized()
        else:
            raise("No normalized data found. Correct the path and filename or set rebuild to True to rebuild the \
                  normalized data.")

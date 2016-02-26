import logging
import time

from invoke import run, task
from time import strftime


__defaultInc__ = 500
__defaultLimit__ = 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@task
def download_source(limit=__defaultInc__, increment=__defaultLimit__, prefix=str(strftime("%Y%m%d%H%M%S"))):
    '''
    Download data from plos api.
    '''
    import sys
    sys.path.append('../../doclabels')
    import json
    import numpy as np
    import csv
    import plosdata
    start = time.clock()
    plosdata.save_source('./data', prefix=prefix, limit=limit, increment=increment)
    print('Source saved. time: {}'.format(time.clock() - start))


@task
def prepare_input(prefix):
    '''
    Prepare input data for scikit-learn classifiers.
    '''



@task
def prepare_cnn_input(prefix):
    '''
    Prepare input data for tensor flow
    '''
    import sys
    sys.path.append('../../doclabels')
    import json
    import numpy as np
    import csv
    import plosdata

    start = time.clock()
    try:
        logging.info('Loading padded docs, responses and vocab...')
        with open('./data/{}-padded'.format(prefix), 'rb') as f:
            paddedreader = csv.reader(f)

            padded = [row for row in paddedreader]
        with open('./data/{}-responses'.format(prefix), 'r') as f2:
            responses = json.load(f2)
        try:
            logging.info('Loading vocabulary...')
            with open('./data/{}-vocab'.format(prefix), 'r') as f3:
                vocab = json.load(f3)
        except:
            logging.info('Missing vocabulary. Creating it using padded docs.')
            vocab = plosdata.build_vocabulary(padded)
            with open('./data/{}-vocab'.format(prefix), 'w') as f3:
                json.dump(vocab, f3)
    except:
        logger.info("Processed files not found. Rebuilding files...")
        # load source files
        filenames = [subject.replace(" ", "_").lower() for subject in plosdata.subject_areas]
        file_list = ['{}/{}-{}.sample'.format('data', prefix, filename) for filename in filenames]
        padded, responses, vocab = plosdata.process_sources(file_list)

        with open('./data/{}-padded'.format(prefix), 'wb') as f:
            paddedwriter = csv.writer(f)
            for i in padded:
                paddedwriter.writerow(i)

        with open('./data/{}-responses'.format(prefix), 'w') as f2:
            json.dump(responses, f2)
        with open('./data/{}-vocab'.format(prefix), 'w') as f3:
            json.dump(vocab, f3)

        logging.info('padded files, responses vocab created. Time: {}'.format(time.clock() - start))

    # Create input data.
    x, y = plosdata.build_input(vocab, padded, responses)
    logging.info('Input files created. Time: {}'.format(time.clock() - start))

    # Save input data to file
    np.savez('./data/{}-input'.format(prefix), x=x, y=y)
    logging.info('Input files saved.')
    end_time = time.clock() - start
    with open('tasks.time', 'a') as ft:
        ft.write("{} input files built complete. Total time: {}".format(prefix, end_time))


@task
def create_embeddings(prefix):
    '''
    Build word embeddings using google news word2vec.
    '''
    import sys
    sys.path.append('../../doclabels')
    import plosdata
    import json
    import numpy as np
    import gensim

    start = time.clock()
    try:
        # load word vec space from file
        logging.info('Try loading word vec space from file...')
        word_vec_space = np.load('./data/{}-word_vec_space.npz'.format(prefix))['arr_0']
        with open('./data/{}-vocab'.format(prefix), 'r') as f3:
                vocab = json.load(f3)
    except:
        logging.warning('word vec space or vocab not found. building word vec space from vocabulary and google word2vec model')
        try:
            logging.info('Loading vocabulary...')
            with open('./data/{}-vocab'.format(prefix), 'r') as f3:
                vocab = json.load(f3)
        except:
            logging.error('Vocabulary not found. Run prepare_cnn_input to build vocabulary')
            raise
        # Load Word2Vec
        model = gensim.models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)

        # Build word vectors space
        word_vec_space = plosdata.build_word_vec_space(vocab, model)
        # Save word_vec_space
        np.savez('./data/{}-word_vec_space'.format(prefix), word_vec_space)

    # Build word vec index
    word_vec_ind = plosdata.build_word_vec_index(vocab, word_vec_space)
    logging.info('Word vec index created. Time: {}'.format(time.clock() - start))

    # Save word vectors index.
    np.savez('./data/{}-word_vec_ind'.format(prefix), word_vec_ind)
    logging.info('Word Vector Index created and saved. Time: {}'.format(time.clock() - start))

    # Create and save embeddings for CNN to use.
    embeddings = plosdata.create_embeddings(word_vec_ind)
    np.savez('./data/{}-embeddings'.format(prefix), embeddings)
    logging.info('Embeddings created and saved. Time: {}'.format(time.clock() - start))
    end_time = time.clock() - start
    with open('tasks.time', 'a') as ft:
        ft.write("{} embeddings built complete. Total time: {}".format(prefix, end_time))


@task
def run_cnn(prefix, split):
    '''
    run cnn_train.py
    arguments: input, embeddings, split (percent of evaluation data)
    '''
    import os
    os.system('python cnn_train.py ./data/{}-input.npz ./data/{}-embeddings.npz {}'.format(prefix, prefix, split))

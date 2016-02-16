import time

from invoke import run, task
from time import strftime


@task
def prepare_cnn_input(limit=500, increment=500, prefix=strftime("%Y%m%d%H%M%S")):
    '''
    Prepare input data for tensor flow
    '''
    import sys
    sys.path.append('../../doclabels')
    import plosdata
    import json
    import numpy as np
    import gensim

    start = time.clock()
    # Process and save source files
    source = plosdata.save_source('./data', prefix=prefix, limit=limit, increment=increment)
    print('Source saved. time: {}'.format(time.clock() - start))
    filenames = [subject.replace(" ", "_").lower() for subject in plosdata.subject_areas]
    file_list = ['{}/{}-{}.sample'.format('data', prefix, filename) for filename in filenames]
    padded, responses, vocab = plosdata.process_sources(file_list)

    with open('{}-padded'.format(prefix), 'w') as f:
        json.dump(padded, f)

    with open('{}-responses'.format(prefix), 'w') as f2:
        json.dump(responses, f2)
    print('padded files, responses vocab created. Time: {}'.format(time.clock() - start))

    # Load Word2Vec
    model = gensim.models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)

    # Build word vectors space
    word_vec_space = plosdata.build_word_vec_space(vocab, model)

    # load word vec space from file
    word_vec_space = np.load('./data/{}-word_vec_space.npy'.format(prefix))

    # Build word vec index
    word_vec_ind = plosdata.build_word_vec_index(vocab, word_vec_space)
    print('Word vec index created. Time: {}'.format(time.clock() - start))
    # Create input data.
    x, y = plosdata.build_input(vocab, padded, responses)

    # Save input data to file

    np.savez('./data/{}-input'.format(prefix), x=x, y=y)

    # Save word vectors index.
    np.savez('./data/{}-word_vec_ind'.format(prefix), word_vec_ind)
    print('Input data created and saved. Time: {}'.format(time.clock() - start))

    # Create and save embeddings for CNN to use.
    embeddings = plosdata.create_embeddings(word_vec_ind)
    np.savez('./data/{}-embeddings'.format(prefix), embeddings)
    print('Embeddings created and saved. Time: {}'.format(time.clock() - start))
    end_time = time.clock() - start
    with open('tasks.time', 'a') as ft:
        ft.write("{}: {}".format(prefix, end_time))


@task
def run_cnn():
    '''
    run cnn_train.py
    '''

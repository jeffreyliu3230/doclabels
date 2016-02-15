
# coding: utf-8

# In[ ]:

import sys
sys.path.append('../../doclabels')
import plosdata
import json
import numpy as np


# In[208]:

reload(plosdata)


# In[202]:

reload(plosapi)


# In[ ]:

plosdata.response_map


# In[165]:

import plosapi
for docs in plosapi.sample("Biology and life sciences", 1000, 500):
    print('haha')


# In[179]:

plosdata.save_source('./data', prefix='testsmall', limit=1000, increment=500)


# In[ ]:

text, response = plosdata.process_source('data/small-people_and_places.sample')


# In[ ]:

filenames = [subject.replace(" ", "_").lower() for subject in plosdata.subject_areas]
file_list = ['{}/{}-{}.sample'.format('data', 'small', filename) for filename in filenames]
padded, responses = plosdata.process_sources(file_list)


# In[ ]:

with open('small-padded', 'w') as f:
    json.dump(padded, f)


# In[ ]:

with open('small-responses', 'w') as f2:
    json.dump(responses, f2)


# In[ ]:

padded, responses= plosdata.load_processed('small-padded', 'small-responses')


# In[ ]:

vocab = plosdata.build_vocabulary(padded)


# In[ ]:

# Load Word2Vec
import gensim
model = gensim.models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[ ]:

# Build word vectors space
word_vec_space = plosdata.build_word_vec_space(vocab, model)


# In[ ]:

# load word vec space from file
word_vec_space = np.load('./data/small-word_vec_space.npy')


# In[ ]:

# Build word vec index
word_vec_ind = plosdata.build_word_vec_index(vocab, word_vec_space)


# In[ ]:




# In[ ]:

x, y = plosdata.build_input(vocab, padded, responses)


# In[ ]:

x.shape


# In[ ]:

np.savez('./data/small-input', x=x, y=y)


# In[ ]:

np.savez('./data/small-word_vec_ind', word_vec_ind)


# In[ ]:

# Modelling
len(word_vec_ind[0])


# In[ ]:

embeddings = plosdata.create_embeddings(word_vec_ind)


# In[ ]:

np.savez('./data/small-embeddings', embeddings)


# In[ ]:

embeddings.shape[1]


# In[ ]:

xandy = np.load('./data/small-embeddings.npz')


# In[ ]:

xandy.files


# In[ ]:

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import plosdata
import logging

from doc_cnn import DocCNN


logger = logging.getLogger(__name__)
logging.basicConfig(filename='cnn_train.log', filemode='w', level=logging.DEBUG)



# In[ ]:

tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")


# In[ ]:

len(word_vec_ind)


# In[ ]:

raw_docs = map(lambda x: " ".join(x).replace(" <PAD/>", ""),padded)


# In[ ]:

raw_docs[1]


# In[ ]:

y_raw = map(lambda x: x.index(1) ,responses)


# In[ ]:

list(shuffle_indices)


# In[ ]:

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_raw)))
x_raw_shuffled = np.asarray(raw_docs)[shuffle_indices]
y_raw_shuffled = np.asarray(y_raw)[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_raw_train, x_raw_test = x_raw_shuffled[:-1000], x_raw_shuffled[-1000:]
y_raw_train, y_raw_test = y_raw_shuffled[:-1000], y_raw_shuffled[-1000:]


# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
X_train_count = vectorizer.fit_transform(list(x_raw_train))


# In[ ]:

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
X_train_tfidf.shape


# In[ ]:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_raw_train)


# In[ ]:

X_new_counts = vectorizer.transform(list(x_raw_test))
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(list(x_raw_test), predicted):
    print('%r => %s' % (doc, y_raw_test[category]))


# In[ ]:

# Prediction accuracy
np.mean(predicted==y_raw_test)


# In[ ]:

# Training accuracy
np.mean(x_raw_test)


# In[ ]:

y_raw_test


# In[213]:

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='squared_hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(list(x_raw_train), y_raw_train)
predicted_svm = text_clf.predict(x_raw_test)
np.mean(predicted_svm == y_raw_test)


# In[115]:

y_raw_train


# In[117]:

text_clf


# In[118]:

# Accuracy on training set
np.mean(text_clf.predict(x_raw_train) == y_raw_train)


# ###### It means more data should be fed to tackle overfitting!

# In[204]:

plosdata.save_source('./data', prefix='large', limit=10000)


# In[209]:

filenames = [subject.replace(" ", "_").lower() for subject in plosdata.subject_areas]
file_list_l = ['{}/{}-{}.sample'.format('data', 'large', filename) for filename in filenames]
padded_l, responses_l, vocab_l = plosdata.process_sources(file_list_l)


# In[210]:

tryload = plosdata.load_source('data/large-biology_and_life_sciences.sample')


# In[ ]:

# now use larger data set

raw_docs_l = map(lambda x: " ".join(x).replace(" <PAD/>", ""),padded_l)
y_raw_l = map(lambda x: x.index(1) ,responses_l)
np.random.seed(10)
shuffle_indices_l = np.random.permutation(np.arange(len(y_raw_l)))
x_raw_shuffled_l = np.asarray(raw_docs_l)[shuffle_indices_l]
y_raw_shuffled_l = np.asarray(y_raw_l)[shuffle_indices_l]
# Split train/test set
x_raw_train_l, x_raw_test_l = x_raw_shuffled_l[:-2000], x_raw_shuffled_l[-2000:]
y_raw_train_l, y_raw_test_l = y_raw_shuffled_l[:-2000], y_raw_shuffled_l[-2000:]


# In[217]:

text_clf_l = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='squared_hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf_l.fit(list(x_raw_train_l), y_raw_train_l)
predicted_svm_l = text_clf_l.predict(x_raw_test_l)
np.mean(predicted_svm_l == y_raw_test_l)


# In[218]:

np.mean(text_clf_l.predict(x_raw_train_l) == y_raw_train_l)


# In[190]:

# get numFound
for subject in plosdata.subject_areas:
    print(plosapi.get_num({'q': 'subject:\"{}\"'.format(subject)}))


# In[ ]:




import logging
import numpy as np
import os
import pickle
import settings
import time

from doclabels.processing.base import MongoProcessor
from doclabels.helpers import generate_response_map, to_classes, OVR_transformer
from pymongo import MongoClient
from random import SystemRandom

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.externals import joblib

from time import strftime

# logging.getLogger().addHandler(logging.StreamHandler())
# logging.basicConfig(filename='log/train.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

random = SystemRandom(settings.DEFAULT_SEED)


def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    # s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


class BatchTrainer():
    """
    Batch train documents classification model.
    The multi-label classification problem will be treated as separate OneVsRest multi-class classification problem
    for each class. Therefore, there will be n_classes classifiers used in the Trainer.
    Performance metrics will be measured for each classifier. An average of all metrics will also be recorded.
    """
    def __init__(self, ids=None, size=None, test_size=settings.DEFAULT_TEST_SIZE, models=None, params=None,
                 run=str(strftime("%Y%m%d%H%M%S")), checkpoint=settings.DEFAULT_CHECKPOINT, cv_folds=0, path='.'):
        self.ids = ids
        self.size = None if ids else size or settings.DEFAULT_SAMPLE_SIZE
        self.test_size = test_size
        self.models = models or {'SGD': SGDClassifier}
        self.params = params or {'SGD': {}}  # To-do: add custome parameters for grid search
        self.run = run
        self.checkpoint = checkpoint
        self.cv_folds = cv_folds
        self.path = path
        self.checkpoints_path = '{}/runs/{}/checkpoints'.format(self.path, self.run)
        self.summaries_path = '{}/runs/{}/summaries'.format(self.path, self.run)
        self.vectorizer_path = '{}/vectorizer'.format(self.checkpoints_path)
        # Models and vectorizer
        self.cls_list = []
        self.vectorizer = None
        # Stats and metrics
        self.cls_stats = {}  # Precision, Recall, F-score, Support
        self.loss = None  # Logloss

    def setup(self, client):
        self.mp = MongoProcessor()
        self.mp.manager.setup(client)

        # Setup paths for storing checkpoints and summaries
        os.system('mkdir -p {}'.format(self.checkpoints_path))
        os.system('mkdir -p {}/train'.format(self.summaries_path))
        os.system('mkdir -p {}/test'.format(self.summaries_path))

        # Get all ids from database.
        if not self.ids:
            self.all_ids = self.mp.fetch_ids()
            self.ids = random.sample(self.all_ids, self.size)
        self.train_ids, self.test_ids = train_test_split(self.ids, test_size=self.test_size)

        print("Training size: {}; test size: {}".format(len(self.train_ids), len(self.test_ids)))
        print("Checkpoint: {}".format(self.checkpoint))

    def train(self, **kwargs):
        """
        Training the data.
        Define arguments for batch_xy and HashingVectorizer in kwargs
        """

        # Initialization
        ngram_range = kwargs.pop('ngram_range', settings.DEFAULT_NGRAM_RANGE)
        stop_words = kwargs.pop('stop_words', 'english')
        total_vect_time = 0
        self.vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                                            non_negative=True, ngram_range=ngram_range, stop_words=stop_words)
        response_map = generate_response_map(settings.SUBJECT_AREAS)
        all_classes = response_map.values()

        # Test data
        (X_test_text, y_test_labels) = self.mp.batch_xy(self.test_ids, batch_size=len(self.test_ids), epochs=1).next()
        X_test = self.vectorizer.transform(map(lambda x: " ".join(x), X_test_text))
        y_test_all = map(lambda x: to_classes(x, response_map), y_test_labels)

        # Create classifiers, and paths for storing classifiers.
        for cls_base_name, cls_base in self.models.items():
            # Create classifiers for all classes
            self.cls_list = [cls_base(**self.params[cls_base_name])] * len(all_classes)
            self.cls_stats[cls_base_name] = {}
            for j in range(len(all_classes)):
                cls_name = cls_base_name + str(j)
                model_path = '{}/{}'.format(self.checkpoints_path, cls_name)
                os.makedirs(model_path)
        # Create path for storing vectorizer.
        os.makedirs(self.vectorizer_path)

        # Main loop.
        for i, (X_train_text, y_train_labels) in enumerate(self.mp.batch_xy(self.train_ids, **kwargs)):
            tick = time.time()
            print('Step {}'.format(i))
            X_train = self.vectorizer.transform(map(lambda x: " ".join(x), X_train_text))
            total_vect_time += time.time() - tick
            print('Vect time: {}'.format(total_vect_time))
            y_train_all = map(lambda x: to_classes(x, response_map), y_train_labels)

            for cls_base_name, cls_base in self.models.items():
                tick = time.time()
                for j in range(len(all_classes)):
                    # Create namespace for storing metrics.
                    cls_name = cls_base_name + str(j)
                    stats = self.cls_stats[cls_base_name][cls_name] = {
                        'n_train': 0, 'n_train_pos': 0,
                        'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
                        'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
                    # transform y
                    y_train = np.asarray(map(lambda x: OVR_transformer(x, j), y_train_all))
                    # import ipdb
                    # ipdb.set_trace()
                    # update estimator with examples in the current mini-batch
                    self.cls_list[j].partial_fit(X_train, y_train, classes=[0, 1])
                    # accumulate test accuracy stats
                    stats['total_fit_time'] += time.time() - tick
                    stats['n_train'] += X_train.shape[0]
                    stats['n_train_pos'] += sum(y_train)
                    tick = time.time()

                    # Metrics
                    stats['accuracy'] = self.cls_list[j].score(X_train, y_train)
                    stats['prediction_time'] = time.time() - tick
                    acc_history = (stats['accuracy'],
                                   stats['n_train'])
                    stats['accuracy_history'].append(acc_history)
                    run_history = (stats['accuracy'],
                                   total_vect_time + stats['total_fit_time'])
                    stats['runtime_history'].append(run_history)

                    with open('{}/train/stats.txt'.format(self.summaries_path), 'a') as f:
                        f.write(progress(cls_name, stats))
                        f.write('\n')

                    # If at a checkpoint, test the model using the test set, and save models and the vectorizer.
                    if (i + 1) % self.checkpoint == 0:
                        print("Step: {}".format(i))
                        y_test = np.asarray(map(lambda x: OVR_transformer(x, j), y_test_all))
                        y_pred = self.cls_list[j].predict(X_test)
                        test_stats = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
                        accuracy = self.cls_list[j].score(X_test, y_test)
                        print("Test metrics for {}: {}".format(cls_name, test_stats))
                        print("Accuracy: {}".format(accuracy))
                        print(confusion_matrix(y_test, y_pred))
                        with open('{}/test/test_stats.txt'.format(self.summaries_path), 'a') as f:
                            f.write(str(test_stats))
                            f.write('\n')
                        # Save the classifiers. Remove previous checkpoints.
                        model_path = '{}/{}'.format(self.checkpoints_path, cls_name)
                        try:
                            os.remove('{}/*'.format(model_path))
                        except:
                            pass
                        joblib.dump(self.cls_list[j], '{}/checkpoint.pkl'.format(model_path), compress=3)
            # Save the vectorizer.
            if (i + 1) % self.checkpoint == 0:
                try:
                    os.remove('{}/*'.format(self.vectorizer_path))
                except:
                    pass
                joblib.dump(self.vectorizer, '{}/checkpoint.pkl'.format(self.vectorizer_path), compress=3)

        # for k in range(len(all_classes)):
        #     y_test = np.asarray(map(lambda x: OVR_transformer(x, k), y_test_all))
        #     test_accuracy = self.cls_list[j].score(X_test, y_test)
        #     print("Test accuracy for {}: {}".format('SGD' + str(k), test_accuracy))
        #     with open('{}/test/test_stats.txt'.format(self.summaries_path), 'a') as f:
        #         f.write(str(test_accuracy))
        #         f.write('\n')

    def save_checkpoint(self):
        """
        Save model at checkpoint.
        """

    def save_stats(self):
        """
        save metrics: Precision, recall, f-score, support, average pricision-score, log loss
        """

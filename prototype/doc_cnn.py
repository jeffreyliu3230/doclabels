import tensorflow as tf
import numpy as np


class DocCNN(object):
    """
    Document classification using CNN.
    Pretrained word vectors, convolution, max-pool, softmax.
    """

    def __init__(self, doc_length, num_classes, embeddings, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.x = tf.placeholder(tf.int32, [None, doc_length], name="x")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.embedding_size = embeddings.shape[1]

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding (using pre-trained word vectors)
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.convert_to_tensor(embeddings, dtype=tf.float32, name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Convolution and maxpool
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cov-maxpool-{}".format(filter_size)):
                # Convolution
                filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, doc_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

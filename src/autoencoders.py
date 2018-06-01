from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, data, labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        pass

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._labels = self.labels[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            labels_rest_part = self.labels[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._labels = self.labels[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]


class AE(object):
    """The class implements the basic auto-encoder."""

    def __init__(self, n_input, n_hidden, tied_w=False, transfer_function=tf.nn.sigmoid, optimizer=tf.train.GradientDescentOptimizer(1.0)):
        # tf.train.GradientDescentOptimizer(0.01)
        """
        The constructor of an auto-encoder.

        Args:
        n_input: `int`, the dimension of input layer.
        n_hidden: `int`, the dimension of hidden layer.
        transfer_function: `Operation`, activation function.
        optimizer: `Operation`, optimizer to train the model.
        """

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        self.weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        if tied_w:
            self.reconstruction = self.transfer(tf.add(tf.matmul(self.hidden, tf.transpose(self.weights['w1'])), self.weights['b2']))
        else:
            self.reconstruction = self.transfer(tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']))

        # cost
        # self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.reconstruction, labels=self.x))

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """Initialize the auto-encoder as soon as we created it.

        The initialization includes w1, b1, w2, b2.

        Returns:
            all_weights: the dictionary containing initialized variables.
        """
        all_weights = dict()

        all_weights['w1'] = tf.Variable(tf.random_normal(shape=[self.n_input, self.n_hidden]), name='w1')
        all_weights['w2'] = tf.Variable(tf.random_normal(shape=[self.n_hidden, self.n_input]), name='w2')

        all_weights['b1'] = tf.Variable(tf.zeros(shape=[self.n_hidden]), name='b1')
        all_weights['b2'] = tf.Variable(tf.zeros(shape=[self.n_input]), name='b2')

        return all_weights

    def train(self, X, epochs=100, batch_size=32, test_size=0.2):
        X_train, X_valid = train_test_split(X, test_size=test_size)
        # Itervatively train.
        for epoch in range(epochs):
            total_batch = int(X_train.shape[0]/batch_size)
            dataset_train = Dataset(X_train, X_train)
            dataset_valid = Dataset(X_valid, X_valid)
            avg_cost_train = 0.0
            avg_cost_valid = 0.0
            for i in range(total_batch):
                batch_x_train, _ = dataset_train.next_batch(batch_size)
                batch_x_valid, _ = dataset_valid.next_batch(batch_size)
                cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: batch_x_train})
                val_cost = self.sess.run(self.cost, feed_dict={self.x: batch_x_valid})
                avg_cost_train += cost/total_batch
                avg_cost_valid += val_cost/total_batch
            print('Epoch ' + str(epoch+1), ': cost =', "{:.3f}".format(avg_cost_train), ' val_cos = ', "{:.3f}".format(avg_cost_valid))
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    def getWeights(self):
        return self.sess.run(self.weights)


def compute_embeddings(X, dim=500):
    num_input = X.shape[1]
    ae = AE(num_input, dim, tied_w=True)
    ae.train(X, epochs=60, batch_size=32)
    features = ae.transform(X)

    return features

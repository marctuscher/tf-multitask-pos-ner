import numpy as np
import os
import tensorflow as tf
import shutil
from random import shuffle

from utilities.keras_progbar import Progbar


class POSModel():

    def __init__(self, embeddings, ntags_pos, ntags_ner, utils):
        """
        Defines the hyperparameters
        """
        # training
        self.ntags_pos = ntags_pos
        self.ntags_ner = ntags_ner
        self.embeddings = embeddings
        self.utils = utils
        self.train_embeddings = False
        self.nepochs =10
        self.keep_prob = 0.5
        self.batch_size = 128
        self.lr_method = "adam"
        self.learning_rate = 0.001
        self.lr_decay = 0.9
        self.clip = -1  # if negative, no clipping
        self.nepoch_no_imprv = 5
        # model hyperparameters
        self.hidden_size_lstm = 300  # lstm on word embeddings
        self.sess = None
        self.saver = None
        # delete ./out so
        if os.path.isdir("./out"):
            shutil.rmtree("./out")
        self.dir_output = "./out"
        self.dir_model = os.getenv("HOME") + str("/tmp/posmodel/model.ckpt")
        self.acc = 0

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, pos_loss, ner_loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_pos_op = optimizer.minimize(pos_loss)
                self.train_ner_op = optimizer.minimize(ner_loss)

                

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.dir_output+"/train",
                                                 self.sess.graph)

    def train(self, train_pos, dev_pos, train_ner, dev_ner):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(self.nepochs):
            print("Epoch {:} out of {:}".format(epoch + 1,
                                                self.nepochs))

            self.run_epoch(train_pos, dev_pos,train_ner, dev_ner, epoch)
            self.learning_rate *= self.lr_decay  # decay learning rate

            if epoch % 2 == 0:

                metrics_pos = self.run_evaluate(dev_ner, not True)
                msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics_pos.items()])
                print(msg)
                # self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=metrics_pos["acc"])]), epoch)
                # early stopping and saving best parameters
                if metrics_pos["acc"]  >= best_score:
                    nepoch_no_imprv = 0
                    self.save_session()
                    best_score = metrics_pos["acc"]
                    print("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.nepoch_no_imprv:
                        print("- early stopping {} epochs without " \
                              "improvement".format(nepoch_no_imprv))
                        break

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        print("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        print(msg)

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        word_ids, sequence_lengths = self.utils.pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if labels is not None:
            labels, _ = self.utils.pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        This is a lookup tensor where each word_id corresponds to an index in this lookup tensor.
        Each index holds a 300-dim vector representing the GloVe word embedding of the word corresponding to this
        word_id in our dictionary
        """
        _word_embeddings = tf.Variable(
            self.embeddings,
            name="word_embeddings_v",
            dtype=tf.float32,
            trainable=self.train_embeddings)

        word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                 self.word_ids, name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """
        Adds the bi-lstm layer and a fully connected layer with softmax output to the graph.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("pos"):
            W_pos = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.hidden_size_lstm, self.ntags_pos])

            b_pos = tf.get_variable("b", shape=[self.ntags_pos],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps_pos = tf.shape(output)[1]
            output_pos = tf.reshape(output, [-1, 2 * self.hidden_size_lstm])
            pred = tf.matmul(output_pos, W_pos) + b_pos
            self.logits_pos = tf.reshape(pred, [-1, nsteps_pos, self.ntags_pos])

        with tf.variable_scope("net"):
            W_ner = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.hidden_size_lstm, self.ntags_ner])

            b_ner = tf.get_variable("b", shape=[self.ntags_ner],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps_ner = tf.shape(output)[1]
            output_ner = tf.reshape(output, [-1, 2 * self.hidden_size_lstm])
            pred_ner = tf.matmul(output_ner, W_ner) + b_ner
            self.logits_ner = tf.reshape(pred_ner, [-1, nsteps_ner, self.ntags_ner])

    def add_pred_op(self):
        """Defines self.labels_pred
        Gets int labels from the output of the softmax layer. The predicted label is
        the argmax of this layer
        """
        self.labels_pred_ner = tf.cast(tf.argmax(self.logits_ner, axis=-1),
                                   tf.int32)
        self.labels_pred_pos = tf.cast(tf.argmax(self.logits_pos, axis=-1),
                                   tf.int32)

    def add_loss_op(self):
        """Losses for training"""
        losses_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits_pos, labels=self.labels)
        mask_pos = tf.sequence_mask(self.sequence_lengths)
        losses_pos = tf.boolean_mask(losses_pos, mask_pos)
        self.loss_pos = tf.reduce_mean(losses_pos)

        losses_ner = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits_ner, labels=self.labels)
        mask_ner = tf.sequence_mask(self.sequence_lengths)
        losses_ner = tf.boolean_mask(losses_ner, mask_ner)
        self.loss_ner = tf.reduce_mean(losses_ner)
        # Scalars for tensorboard
        tf.summary.scalar("loss", self.loss_pos)
        tf.summary.scalar("loss", self.loss_ner)
    def build(self):
        """
        Build the computational graph with functions defined earlier
        """
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(self.lr_method, self.lr, self.loss_pos, self.loss_ner,
                          self.clip)
        self.initialize_session()

    def predict_batch_ner(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        Predict a batch of sentences (list of word_ids)
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        labels_pred = self.sess.run(self.labels_pred_ner, feed_dict=fd)
        return labels_pred, sequence_lengths

    def predict_batch_pos(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        Predict a batch of sentences (list of word_ids)
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        labels_pred = self.sess.run(self.labels_pred_pos, feed_dict=fd)
        return labels_pred, sequence_lengths

    def run_epoch(self, train_pos, dev_pos, train_ner, dev_ner, epoch):
        """Performs one complete epoch over the dataset

        Args:
            train: dataset for training that yields tuple of sentences, tags
            dev: dataset for evaluation that yields tuple of sentences, tags
            epoch: (int) index of the current epoch

        Returns:
            acc: (float) current accuracy score over evaluation dataset

        """
        # progbar stuff for logging
        batch_size = self.batch_size
        nbatches = (len(train_pos) + batch_size - 1)+(len(train_ner)+batch_size-1) // batch_size
        prog = Progbar(target=nbatches)
        shuffle(train_pos)
        shuffle(train_ner)
        for i, (words, labels, state) in enumerate(self.utils.mixed_minibatches(train_pos, train_ner, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.learning_rate,
                                       self.keep_prob)
            if state =='pos':
                _, train_loss, summary = self.sess.run(
                    [self.train_pos_op, self.loss_pos, self.merged], feed_dict=fd)

                prog.update(i + 1, [("train loss", train_loss)])

            else:
                _, train_loss, summary = self.sess.run(
                    [self.train_ner_op, self.loss_ner, self.merged], feed_dict=fd)

                prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)


    def run_evaluate(self, test, pos):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        for words, labels in self.utils.minibatches(test, self.batch_size):
            if pos:
                labels_pred, sequence_lengths = self.predict_batch_pos(words)
            else:
                labels_pred, sequence_lengths = self.predict_batch_ner(words)
            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
        acc = np.mean(accs)
        # set self.acc for Tensorboard visualization
        return {"acc": 100 * acc}


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

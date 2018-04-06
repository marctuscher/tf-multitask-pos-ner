import numpy as np
import os
import tensorflow as tf
import shutil

from utilities.keras_progbar import Progbar


class NERModel():

    def __init__(self, embeddings, ntags, utils):
        """
        Defines the hyperparameters
        """
        # training
        self.ntags = ntags
        self.embeddings = embeddings
        self.utils = utils
        self.train_embeddings = False
        self.nepochs =25
        self.keep_prob = 0.8
        self.batch_size = 1024
        self.lr_method = "adam"
        self.learning_rate = 0.01
        self.lr_decay = 0.9
        self.clip = 1  # if negative, no clipping
        self.nepoch_no_imprv = 5
        # model hyperparameters
        self.hidden_size_lstm = 600  # lstm on word embeddings
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

    def add_train_op(self, lr_method, lr, loss, clip=-1):
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
                self.train_op = optimizer.minimize(loss)

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

    def train(self, train, dev, inv_classes):
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

            self.run_epoch(train, dev, epoch)
            self.learning_rate *= self.lr_decay  # decay learning rate

            if epoch % 2 == 0:

                metrics = self.run_evaluate(dev, inv_classes)
                msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
                print(msg)
                self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=metrics["acc"])]), epoch)
                # early stopping and saving best parameters
                if metrics["acc"]  >= best_score:
                    nepoch_no_imprv = 0
                    self.save_session()
                    best_score = metrics["acc"]
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

            # output = tf.concat([output_fw, output_bw], axis=-1)
            # output = tf.nn.dropout(output, self.dropout)
            output = tf.add(output_fw, output_bw)
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[self.hidden_size_lstm, self.ntags])

            b = tf.get_variable("b", shape=[self.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, self.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            pred = tf.nn.dropout(pred, self.dropout)
            self.logits = tf.reshape(pred, [-1, nsteps, self.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred
        Gets int labels from the output of the softmax layer. The predicted label is
        the argmax of this layer
        """
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                   tf.int32)

    def add_loss_op(self):
        """Losses for training"""
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        # Scalars for tensorboard
        tf.summary.scalar("loss", self.loss)
    def build(self):
        """
        Build the computational graph with functions defined earlier
        """
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(self.lr_method, self.lr, self.loss,
                          self.clip)
        self.initialize_session()

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        Predict a batch of sentences (list of word_ids)
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
        return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
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
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        for i, (words, labels) in enumerate(self.utils.minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.learning_rate,
                                       self.keep_prob)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)


    def run_evaluate(self, test, inv_classes):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in self.utils.minibatches(test, self.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)
            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                acc = [a == b for (a, b) in zip(lab, lab_pred)]
                accs += acc

                lab_chunks      = set(self.utils.get_chunks(lab, inv_classes))
                lab_pred_chunks = set(self.utils.get_chunks(lab_pred,
                                                        inv_classes))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
        acc = np.mean(accs)
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return {"acc": 100 * acc, "f1": f1}


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

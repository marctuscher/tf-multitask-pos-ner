import imp
import os

import gensim
import numpy as np
import tensorflow as tf

from playground.parsing import extend_parser as parser
from playground.utilities import utils

imp.reload(parser)
imp.reload(utils)
"""
Sieht so aus als würde das mal laufen. Hab jegliches evaluieren allerdings auskommentiert.
Müssen wir die Tage noch machen. Ausserdem habe ich das gefühl dass mit den classes was nicht
stimmt. Ihr müsst euch mal das classes dic anschauen und die class_to_int methode in utils
"""

# TODO Model abspeichern. ausserdem müssten wir auch nicht das komplette model laden sondern könnten das trimmen.
model = gensim.models.KeyedVectors.load_word2vec_format("/home/marc/Downloads/glove.840B.300d/glove.840B.300d.w2vformat.txt", binary=False)
print("Imported")
sentences_train, lexikon_vec, lexikon_dic, classes = parser.parse_pos_training(os.getenv("DATA_DIR_DL")+'/pos/en-dev.txt', model)
print("Most similar to dog according to embedding:")
print(model.most_similar("dog"))


int_classes, classes_dic = utils.classes_to_int(classes)
dim_words = 300
# das mit den graphs hab ich noch nicht ganz raus. man muss den graph für die interactive session immer
# wieder flushen wenn man das wieder startet. klappt aber irgendwie nicht immer
embeddings = np.zeros([len(lexikon_dic), 300])
# hier kannste nachschauen was ne id fürn vector hat. kannst eigentlich auch über das model
# aber der dude aus dem tutorial hat das glaub mit so nem tensorflow lookup gemacht. soll woll besser sein
# seht ihr weiter unten
for word in lexikon_dic:
    word_idx = lexikon_dic[word]
    embeddings[word_idx] = np.asarray(model[word])

ntags = 0
hidden_size = 300
learning_rate = 0.001
batch_size = 16
# hier mach ich paar placeholders, die müssen zu den keys in feed_dict passen
# hier resette ich den graph, funktioniert halt nicht immer :D
tf.reset_default_graph()
tmp_word_embeddings = tf.Variable(
        embeddings,
        name="tmp_word_embeddings",
        dtype=tf.float32,
        trainable=False)


word_ids = tf.placeholder(tf.int32, shape=[None, None],
                              name="word_ids")
word_embeddings = tf.nn.embedding_lookup(tmp_word_embeddings,
                                         word_ids, name="word_embeddings")

sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                       name="sequence_lengths")
word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                   name="word_lengths")
labels = tf.placeholder(tf.int32, shape=[None, None],
                             name="labels")
dropout = tf.placeholder(dtype=tf.float32, shape=[],
                             name="dropout")
# hier ist der kontext bi-lstm, die sequence lengths sind in den batches variabl
with tf.variable_scope("bi-lstm"):
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, word_embeddings,
        sequence_length=sequence_lengths, dtype=tf.float32)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.nn.dropout(output, dropout)

# fully connected layer für classification, hab das einfach übernommen
with tf.variable_scope("proj"):
    W = tf.get_variable("W", dtype=tf.float32,
                        shape=[2*hidden_size, len(classes)])
    b = tf.get_variable("b", shape=[len(classes)],
                        dtype=tf.float32, initializer=tf.zeros_initializer())

    nsteps = tf.shape(output)[1]
    output = tf.reshape(output, [-1, 2*hidden_size])
    pred = tf.matmul(output, W) + b
    logits = tf.reshape(pred, [-1, nsteps, len(classes)])

labels_pred = tf.cast(tf.argmax(logits, axis=-1),
                           tf.int32)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
mask = tf.sequence_mask(sequence_lengths)
losses = tf.boolean_mask(losses, mask)
loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
print("Initializing tf session")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

data = utils.sen_dict_to_tuple(sentences_train, lexikon_dic, classes_dic)
n_epochs = 1000
# feed dict ist ne ziemliche zicke, hab ewig gebraucht bis ich das am laufen hatte...
# nicht über die variablen wundern
def get_feed_dict(words, labels_l=None, lr=None, dropout_=None):
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
    # perform padding of the given data
    words, sequence_l = utils.pad_sequences(words, 0)
        # build feed dictionary
    feed = {
        word_ids: words,
        sequence_lengths: sequence_l
    }

    if labels is not None:
        labels_, _ = utils.pad_sequences(labels_l, -1)
        feed[labels] = labels_

    if dropout is not None:
        feed[dropout] = dropout_
    return feed, sequence_lengths

def run_epoch(train, test, epoch):
    batch_size= 16
    """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
    # progbar stuff for logging
    batch_size = batch_size
    nbatches = (len(train) + batch_size - 1) // batch_size
    prog = utils.Progbar(target=nbatches)

    # iterate over dataset
    for i, (words, labels) in enumerate(utils.minibatches(train, batch_size)):
        fd, _ = get_feed_dict(words, labels, learning_rate,
                                   dropout_=0.5)

        _, train_loss= sess.run(
            [train_op, loss], feed_dict=fd)

        prog.update(i + 1, [("train loss", train_loss)])

        # tensorboard
        # if i % 10 == 0:
            # self.file_writer.add_summary(summary, epoch*nbatches + i)

    # metrics = self.run_evaluate(dev)
    # msg = " - ".join(["{} {:04.2f}".format(k, v)
    #             for k, v in metrics.items()])
    # print(msg)

    # return metrics["f1"]


for epoch in range(n_epochs):
    run_epoch(data, None, epoch)

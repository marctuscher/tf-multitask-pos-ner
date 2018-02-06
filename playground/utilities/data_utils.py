import pickle
import gensim
import numpy as np
import time

class Utils:

    def __init__(self):
        """
        Constructor...
        """


    def load_glove_txt(self, filename):
        """
        Load the glove model from a given path
        """
        start = time.time()
        print("Started importing Glove Model from textfile")
        self.glove = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)
        print("Finished importing Glove Model in ", start - time.time(), " seconds")

    def save_glove_pkl(self, filename):
        """
        Save the pretrained glove embeddings to pkl file
        """
        with open(filename, 'wb') as output:
            print("Started saving embeddings to binary")
            pickle.dump(self.glove, output, pickle.HIGHEST_PROTOCOL)
            print("Finished")


    def load_glove_pkl(self, filename):
        """
        Load pretrained glove embeddings from file
        Always do this first, since importing the sentences depends on a lookup in GloVe Model
        """
        start = time.time()
        with open(filename, 'rb') as inp:
            print("Started importing Glove Model from binary file")
            self.glove = pickle.load(inp)
            print("Imported binary embeddings in ", time.time()-start, " seconds!")


    def parse_pos(self, filename):
        """
        pos:
        POS tags are according to the Penn Treebank POS tagset: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        format:
        word \t tag
        One word per line, sentences separated by newline.

        Parsing to an array of dicts, maybe not the best solution
        """
        start = time.time()
        print("Started loading sentences form textfile")
        sentences = []
        tmpdic = {'words': [], 'tags':[], 'wc': 0}
        dictionary_dic = {}
        index = 0
        classes_dic = {}
        classindex = 0
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if line != "\n":
                    word, tag = line.split()
                    if word in self.glove.wv.vocab:
                        tmpdic['words'].append(word)
                        if word not in dictionary_dic.keys():
                            dictionary_dic[word]= index
                            index+=1
                        if tag not in classes_dic.keys():
                            classes_dic[tag] = classindex
                            classindex+=1
                        tmpdic['tags'].append(tag)
                else:
                    tmpdic['wc'] = len(tmpdic['words'])
                    for key in tmpdic.keys():
                        tmpdic[key]= np.array(tmpdic[key])
                    if tmpdic['wc'] == 0:
                        tmpdic = {'words': [], 'tags':[], 'wc': 0}
                    else:
                        sentences.append(tmpdic)
                        tmpdic = {'words': [], 'tags':[], 'wc': 0}
        print("Imported sentences in ", time.time()-start, " seconds")
        return sentences, dictionary_dic, classes_dic


    def _pad_sequences(self,sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length


    def pad_sequences(self, sequences, pad_tok, nlevels=1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids
        Returns:
            a list of list where each sublist has same length
        """
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = self._pad_sequences(sequences,
                                                pad_tok, max_length)
        return sequence_padded, sequence_length


    def words2ids (self, sen, dictionary):
        """
        Convert a list of words into a list of ids according to vocab
        """
        l = []
        for word in sen:
            l.append(dictionary[word])
        return l


    def tags_to_int(self, tags, classes_dic):
        """
        Convert class tags to int
        """
        l = list()
        for tag in tags:
            l.append(classes_dic[tag])
        return l


    def sen_dict_to_tuple(self, sentences, dictionary, classes_dic):
        """
        Returns a list of tuples (sentences, tags)
        """
        l = []
        for sen in sentences:
            tmp = self.words2ids(sen["words"], dictionary)
            tags = self.tags_to_int(sen["tags"], classes_dic)
            l.append((tmp, tags))
        return l


    def minibatches(self, data, minibatch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
        Yields:
            list of tuples
        """
        x_batch, y_batch = [], []
        for (x, y) in data:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []

            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]

        if len(x_batch) != 0:
            yield x_batch, y_batch


util = Utils()
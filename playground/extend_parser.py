import string
import pandas as pd
import os
import numpy as np
from numba import jit
import tensorflow as tf

def parse_pos(filename):
    """
    pos:
    POS tags are according to the Penn Treebank POS tagset: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    format:
    word \t tag
    One word per line, sentences separated by newline.

    Parsing to an array of dicts, maybe not the best solution
    """
    sentences = []
    tmpdic = {'words': [], 'tags':[], 'wc': 0}
    dictionary = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line != "\n":
                word, tag = line.split()
                # TODO ugly hard coded shit, sorry for that :) should we change it?
                if word not in string.punctuation and word != '``' and word != '\'\'' and word!= '-rrb-':
                    word = str(word.lower())
                    tmpdic['words'].append(word)
                    if word not in dictionary:
                        dictionary.append(word)
                    tmpdic['tags'].append(tag)
            else:
                tmpdic['wc'] = len(tmpdic['words'])
                for key in tmpdic.keys():
                    tmpdic[key]= np.array(tmpdic[key])
                sentences.append(tmpdic)
                tmpdic = {'words': [], 'tags':[], 'wc': 0}
    return sentences, np.array(dictionary)


def to_dataframe(sentences_as_dict):
    """
    Proposed method for storing the data: in Pandas Dataframe. However
    pandas Dataframe are not best-suited for arrays over multivariate time-series
    data, which is often the case in nlp tasks
    """
    return pd.DataFrame(sentences_as_dict)


@jit
def get_dictionary_np(sentences):
    dictionary = np.empty(0)
    count = 0
    size = sentences['words'].shape[0]
    for arr in sentences['words'].values:
        print(count/size)
        count+=1
        dictionary = np.hstack((dictionary, arr))
    return dictionary.unique()
    # TODO this would be nice: return sentences['words'].values.flatten().unique()
    # but pandas support for 3d arrays is shit

@jit
def get_dictionary_tf(sentences):
    """
    Not working, getting conversion error somewhere. We first need to find a numerical interpretation
    """
    dictionary = tf.constant([''])
    count = 0
    for arr in sentences['words'].values:
        print(count/sentences['words'].shape[0])
        count+=1
        dictionary= tf.concat([dictionary, tf.constant(arr)], axis=0)
    return dictionary



def main():
    # TODO check for other shitty words in data
    sentences_train, dict1 = parse_pos(os.getenv('DATA_DIR_DL')+'/pos/en-train.txt')
    sentences_train = to_dataframe(sentences_train)
    #print(sentences_train.shape)
    #np.save(os.getenv('DATA_DIR_DL')+'/pos/nparray_dictionary.npy', dict1)
    # TODO check if dictionary is unique
    dictionary = np.load(os.getenv('DATA_DIR_DL')+'/pos/nparray_dictionary.npy')
    for word in dictionary:
        print (word)
if __name__ == '__main__':
    main()

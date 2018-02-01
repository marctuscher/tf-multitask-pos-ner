import string
import pandas as pd
import os
import numpy as np
from numba import jit
import tensorflow as tf
import pickle



def parse_pos_training(filename, model):
    """
    pos:
    POS tags are according to the Penn Treebank POS tagset: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    format:
    word \t tag
    One word per line, sentences separated by newline.

    Parsing to an array of dicts, maybe not the best solution
    """
    print("reloeaded parser")
    sentences = []
    tmpdic = {'words': [], 'tags':[], 'wc': 0}
    dictionary_vec = []
    dictionary_dic = {}
    index = 0
    classes_dic = {}
    classindex = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line != "\n":
                word, tag = line.split()
                # TODO ugly hard coded shit, sorry for that :) should we change it?
                if word in model.wv.vocab:
                    word = str(word.lower())
                    tmpdic['words'].append(word)
                    if word not in dictionary_vec:
                        dictionary_vec.append(word)
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
                sentences.append(tmpdic)
                tmpdic = {'words': [], 'tags':[], 'wc': 0}
    return sentences, np.array(dictionary_vec), dictionary_dic, classes_dic


def parse_pos_validation(filename):
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
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line != "\n":
                word, tag = line.split()
                # TODO ugly hard coded shit, sorry for that :) should we change it?
                if word not in string.punctuation and word != '``' and word != '\'\'' and word!= '-rrb-':
                    word = str(word.lower())
                    tmpdic['words'].append(word)
                    tmpdic['tags'].append(tag)
            else:
                tmpdic['wc'] = len(tmpdic['words'])
                for key in tmpdic.keys():
                    tmpdic[key]= np.array(tmpdic[key])
                sentences.append(tmpdic)
                tmpdic = {'words': [], 'tags':[], 'wc': 0}
    return sentences


def to_dataframe(sentences_as_dict):
    """
    Proposed method for storing the data: in Pandas Dataframe. However
    pandas Dataframe are not best-suited for arrays over multivariate time-series
    data, which is often the case in nlp tasks
    """
    return pd.DataFrame(sentences_as_dict)



def import_data(filename):
    sentences_train, dic_vec, dic_dict = parse_pos_training(filename)
    sentences_train = to_dataframe(sentences_train)
    return sentences_train, dic_vec, dic_dict



def save_data(out_dir, sentences, lexikon_vec, lexikon_dic):
    print("Importing finished, now saving")

    np.save(out_dir+'nparray_dictionary.npy', lexikon_vec)
    with open(out_dir+'lexikon_dic.p', 'wb') as fp:
        pickle.dump(lexikon_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)
    sentences.to_pickle(out_dir+'dataframe.p')

    print('Finished saving')

def load_data(in_dir):

    print('Start loading data')

    dic_vec = np.load(in_dir+'nparray_dictionary.npy')
    sentences = pd.read_pickle(in_dir+'dataframe.p')
    dic_dic = pickle.load(open(in_dir+'lexikon_dic.p', 'rb'))

    print('finished this')
    return sentences, dic_vec, dic_dic

@jit
def inverted_onehot(sentences, lexikon_dic, maxwc):
    size = len(sentences)
    sentences = np.array(sentences)
    count = 0
    X = []
    for sen in sentences:
        arr = np.zeros(maxwc)
        j = 0
        print(count/size)
        for word in sen['words']:
            arr[j] = lexikon_dic[word]
            j+=1
        X.append(arr)
        count+=1
    return X

@jit
def get_y(sentences, classes):
    y = []
    classcount = len(classes.keys())
    for sen in sentences:
        ind = 0
        arr = np.zeros(len(sen['tags']))
        for tag in sen['tags']:
            arr[ind] = classes[tag]
            ind +=1
        y.append(arr)
    return y






def main():
    data_dir = os.getenv('DATA_DIR_DL')
    print ("data_dir: ", data_dir)
    sentences_train, lexikon_vec, lexikon_dic, classes = parse_pos_training(data_dir+'/pos/en-dev.txt')
    maxwc = 125
    X_train = inverted_onehot(sentences_train, lexikon_dic, maxwc)
    y_train = get_y(sentences_train, classes)

if __name__ == '__main__':
    main()

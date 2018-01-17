import os
import numpy as np
import collections
from pprint import pprint

def parse_pos_training(filename):
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
    dictionary_vec = []
    classes_dic ={}
    classindex = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line != "\n":
                word, tag = line.split()
                # TODO ugly hard coded shit, sorry for that :) should we change it?
                tmpdic['words'].append(word)
                if word not in dictionary_vec:
                    dictionary_vec.append(word)
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
    return sentences, np.array(dictionary_vec), classes_dic


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
        unk_count += 1
    data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def main():
    data_dir = os.getenv('DATA_DIR_DL')
    sentences, words, classes = parse_pos_training(data_dir+'/pos/en-dev.txt')
    data, count, dictionary, reversed_dictionary = build_dataset(words, words.shape[0])
    pprint(dictionary)


if __name__ == '__main__':
    main()

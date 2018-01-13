import string
import pandas as pd
import os

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
    tmpdic = {'words': [], 'tags':[]}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line != "\n":
                word, tag = line.split()
                # TODO ugly hard coded shit, sorry for that :) should we change it?
                if word not in string.punctuation and word != '``' and word != '\'\'':
                    tmpdic['words'].append(word.lower())
                    tmpdic['tags'].append(tag.lower())
            else:
                sentences.append(tmpdic)
                tmpdic = {'words': [], 'tags':[]}
    return sentences


def to_dataframe(sentences):
    """
    Proposed method for storing the data: in Pandas Dataframe. However
    pandas Dataframe are not best-suited for arrays over multivariate time-series
    data, which is often the case in nlp tasks
    """
    return pd.DataFrame(sentences)

def main():
    # TODO check for other shitty words in data
    sentences = to_dataframe(parse_pos(os.getenv('DATA_DIR_DL')+'/pos/en-train.txt'))
    print (sentences.loc[0, 'words'][0])

if __name__ == '__main__':
    main()

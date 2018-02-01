import gensim
import numpy as np
import os,sys
from gensim.models import Word2Vec
import extend_parser as parser
import imp

imp.reload(parser)

# model = gensim.models.KeyedVectors.load_word2vec_format("/Users/marc/Downloads/glove.840B.300d.w2vformat.txt", binary=False)
print("Imported")
sentences_train, lexikon_vec, lexikon_dic, classes = parser.parse_pos_training('/Users/marc/Nextcloud/deep_learning/pos/en-dev.txt', model)
# print(model.most_similar("dog"))
for w in lexikon_vec:
    model.wv.vocab[w]

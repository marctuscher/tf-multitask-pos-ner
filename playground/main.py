from utilities.data_utils import Utils
import os
from networks.pos_bilstm import POSModel
import numpy as np

util = Utils()

util.load_glove_pkl(os.getenv("DATA_DIR_DL")+str('/word2vec/glove.pkl'))
# util.load_glove_txt(os.getenv("DATA_DIR_DL")+str('/word2vec/glove.840B.300d.w2vformat.txt'))
sentences_train, lexikon_dic, classes = util.parse_pos(os.getenv("DATA_DIR_DL")+'/pos/en-dev.txt')
sentences_val, lexikon_val_dic, classes_val = util.parse_pos(os.getenv("DATA_DIR_DL")+'/pos/en-test.txt')

print(sentences_train[1])
print(sentences_train[3])
embeddings = np.zeros([len(lexikon_dic), 300])
for word in lexikon_dic:
    word_idx = lexikon_dic[word]
    embeddings[word_idx] = np.asarray(util.glove[word])

count = 0
dev = util.sen_dict_to_tuple(sentences_train, lexikon_dic, classes)
val = util.sen_dict_to_tuple(sentences_val, lexikon_val_dic, classes_val)
pos = POSModel(embeddings, len(classes), util)
pos.build()

pos.train(dev, val)

print(pos.predict_batch([val[0][0]]))
print(val[0][1])


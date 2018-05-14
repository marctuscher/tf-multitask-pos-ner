from core.utilities.data_utils import util
import os
from core.networks.add_multi_task import MultiTaskModel
import time
import sys

def main():

    if sys.argv[1].endswith('.pkl'):
        util.load_glove_pkl(sys.argv[1])
    else:
        util.load_glove_txt(util.load_glove_txt(sys.argv[1]))
    sentences_pos_train, lexicon_pos_dic, classes_pos = util.parse_pos(sys.argv[2])
    sentences_pos_val, lexikon_val_pos_dic, classes_val = util.parse_pos(sys.argv[3])

    sentences_ner_train, lexicon_ner_dic, classes_ner = util.parse_ner(sys.argv[4])
    sentences_ner_val, lexikon_val_ner_dic, classes_val = util.parse_ner(sys.argv[5])
    dir_model = sys.argv[6]
    util.save_glove_pkl(str(dir_model)+"glove.pkl")
    util.save_classes_to_json(classes_pos, str(dir_model)+"classes_pos.json")
    util.save_classes_to_json(classes_ner, str(dir_model)+"classes_ner.json")
    # same words should have same ids, merging the two dictionaries
    dictionary, embeddings = util.generate_embeddings([lexicon_pos_dic, lexikon_val_pos_dic, lexicon_ner_dic, lexikon_val_ner_dic])
    # saving embeddings for webserver
    util.save_embeddings(embeddings, str(dir_model)+"embeddings.npy")

    util.save_classes_to_json(dictionary, str(dir_model)+"dictionary.json")
    dev_pos = util.sen_dict_to_tuple(sentences_pos_train, dictionary, classes_pos)
    val_pos = util.sen_dict_to_tuple(sentences_pos_val, dictionary, classes_pos)

    dev_ner = util.sen_dict_to_tuple(sentences_ner_train, dictionary, classes_ner)
    val_ner = util.sen_dict_to_tuple(sentences_ner_val, dictionary, classes_ner)
    multi = MultiTaskModel(len(classes_pos), len(classes_ner), util, dir_model)
    multi.build(embeddings)
    start = time.time()
    multi.train(dev_pos, val_pos, classes_pos,  dev_ner, val_ner, classes_ner)
    print("finished after", time.time()-start)

if __name__ == '__main__':
    main()


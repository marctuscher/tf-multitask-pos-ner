from utilities.data_utils import util
import os
from networks.add_multi_task import MultiTaskModel
import numpy as np

def main():

    util.load_glove_pkl(os.getenv("DATA_DIR_DL")+str('/word2vec/glove.pkl'))
    # util.load_glove_txt(os.getenv("DATA_DIR_DL")+str('/word2vec/glove.840B.300d.w2vformat.txt'))
    sentences_pos_train, lexicon_pos_dic, classes_pos = util.parse_pos(os.getenv("DATA_DIR_DL")+'/pos/en-train.txt')
    sentences_pos_val, lexikon_val_pos_dic, classes_val = util.parse_pos(os.getenv("DATA_DIR_DL")+'/pos/en-dev.txt')

    sentences_ner_train, lexicon_ner_dic, classes_ner = util.parse_ner(os.getenv("DATA_DIR_DL")+'/ner/train.iob')
    sentences_ner_val, lexikon_val_ner_dic, classes_val = util.parse_ner(os.getenv("DATA_DIR_DL")+'/ner/dev.iob')
    # same words should have same ids, merging the two dictionaries
    dictionary, embeddings = util.generate_embeddings([lexicon_pos_dic, lexikon_val_pos_dic, lexicon_ner_dic, lexikon_val_ner_dic])

    dev_pos = util.sen_dict_to_tuple(sentences_pos_train, dictionary, classes_pos)
    val_pos = util.sen_dict_to_tuple(sentences_pos_val, dictionary, classes_pos)

    dev_ner = util.sen_dict_to_tuple(sentences_ner_train, dictionary, classes_ner)
    val_ner = util.sen_dict_to_tuple(sentences_ner_val, dictionary, classes_ner)
    multi = MultiTaskModel(embeddings, len(classes_pos), len(classes_ner), util)
    inv_classes_ner = {idx: tag for tag, idx in classes_ner.items()}
    inv_classes_pos = {idx: tag for tag, idx in classes_pos.items()}
    multi.build()
    multi.train(dev_pos, val_pos, classes_pos,  dev_ner, val_ner, classes_ner)

    # invert class dict (idx as key,tag as value)

    sentence = ' '.join(sentences_ner_val[0]['words'])
    print("test sentence: ", sentence)
    predicted_tags_idxs = pos.predict_batch_ner([val_ner[0][0]])
    predicted_tags = [inv_classes_ner[tag_idx] for tag_idx in predicted_tags_idxs[0][0].tolist()]
    print("predicted tags: ", predicted_tags)
    #correct_tags_idxs = val[0][1]
    #correct_tags = [inv_classes[tag_idx] for tag_idx in correct_tags_idxs]
    print("correct tags: ", sentences_ner_val[0]['tags'])


if __name__ == '__main__':
    main()


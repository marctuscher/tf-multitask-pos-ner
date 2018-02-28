from flask import Flask, request
import json
app = Flask(__name__, static_url_path='')
from utilities.data_utils import util
import os
from networks.add_multi_task import POSModel

util.load_glove_pkl(os.getenv("DATA_DIR_DL")+str('/word2vec/glove.pkl'))
# util.load_glove_txt(os.getenv("DATA_DIR_DL")+str('/word2vec/glove.840B.300d.w2vformat.txt'))
sentences_pos_train, lexicon_pos_dic, classes_pos = util.parse_pos(os.getenv("DATA_DIR_DL")+'/pos/en-train.txt')
sentences_pos_val, lexikon_val_pos_dic, classes_val = util.parse_pos(os.getenv("DATA_DIR_DL")+'/pos/en-test.txt')

sentences_ner_train, lexicon_ner_dic, classes_ner = util.parse_ner(os.getenv("DATA_DIR_DL")+'/ner/train.iob')
sentences_ner_val, lexikon_val_ner_dic, classes_val = util.parse_ner(os.getenv("DATA_DIR_DL")+'/ner/test.iob')
# same words should have same ids, merging the two dictionaries
dictionary, embeddings = util.generate_embeddings([lexicon_pos_dic, lexikon_val_pos_dic, lexicon_ner_dic, lexikon_val_ner_dic])

dev_pos = util.sen_dict_to_tuple(sentences_pos_train, dictionary, classes_pos)
val_pos = util.sen_dict_to_tuple(sentences_pos_val, dictionary, classes_pos)

dev_ner = util.sen_dict_to_tuple(sentences_ner_train, dictionary, classes_ner)
val_ner = util.sen_dict_to_tuple(sentences_ner_val, dictionary, classes_ner)
print([val_ner[0][0]])
pos = POSModel(embeddings, len(classes_pos), len(classes_ner), util)
pos.build()
pos.train(dev_pos, val_pos, dev_ner, dev_ner)

inv_classes_ner = {idx: tag for tag, idx in classes_ner.items()}
inv_classes_pos = {idx: tag for tag, idx in classes_pos.items()}


@app.route('/<path:path>')
def send(path):
    return app.send_from_directory("", path)

@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("is called prdeict")
    sen = util.split_sentence(request.data.decode())
    sen_final = {"words":[]}
    for w in sen:
        w = str(w)
        if w in dictionary:
            sen_final["words"].append(w)
    print (sen_final)
    test = util.sen_dict_to_tuple_pred([sen_final], dictionary)
    predicted_tags_idxs_ner = pos.predict_batch_ner([test[0][0]])
    predicted_tags_idxs_pos = pos.predict_batch_pos([test[0][0]])
    sen_final["tags_ner"] = [inv_classes_ner[tag_idx] for tag_idx in predicted_tags_idxs_ner[0][0].tolist()]
    sen_final["tags_pos"] = [inv_classes_pos[tag_idx] for tag_idx in predicted_tags_idxs_pos[0][0].tolist()]
    return json.dumps(sen_final)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

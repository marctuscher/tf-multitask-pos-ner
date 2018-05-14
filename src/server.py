from flask import Flask, request
import json
app = Flask(__name__, static_url_path='')
from core.utilities.data_utils import util
import os
from core.networks.add_multi_task import MultiTaskModel
import sys


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
    predicted_tags_idxs_ner = multi.predict_batch_ner([test[0][0]])
    predicted_tags_idxs_pos = multi.predict_batch_pos([test[0][0]])
    sen_final["tags_ner"] = [inv_classes_ner[tag_idx] for tag_idx in predicted_tags_idxs_ner[0][0].tolist()]
    sen_final["tags_pos"] = [inv_classes_pos[tag_idx] for tag_idx in predicted_tags_idxs_pos[0][0].tolist()]
    return json.dumps(sen_final)

if __name__ == "__main__":
    dir_model = sys.argv[1]
    util.load_glove_pkl(str(dir_model)+'/glove.pkl')
    dictionary = util.load_classes_from_json(str(dir_model)+"dictionary.json")
    classes_pos = util.load_classes_from_json(str(dir_model)+"classes_pos.json")
    classes_ner = util.load_classes_from_json(str(dir_model)+"classes_ner.json")
    embeddings = util.load_embeddings(str(dir_model)+"embeddings.npy")
    inv_classes_ner = {idx: tag for tag, idx in classes_ner.items()}
    inv_classes_pos = {idx: tag for tag, idx in classes_pos.items()}

    multi = MultiTaskModel(len(classes_pos), len(classes_ner), util, dir_model)
    multi.build(embeddings)
    multi.restore_session()
    app.run(host='0.0.0.0', port=8080)

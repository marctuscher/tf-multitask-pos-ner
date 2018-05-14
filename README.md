# tf-multitask-pos-ner
Deep multi-task learning for POS-tagging and NER using tensorflow. 
## Training the Model
To train the model, a version of GloVe is required. Please download it here http://nlp.stanford.edu/data/glove.840B.300d.zip and unzip it. Additionally, you have to provide training and validation data for both, POS-Tagging and NER. Please see sample-data for the format of the files. Training is started by:
```
cd src/
mkdir model/
python3 main.py /path/to/glove.txt /path/to/pos_train.txt /path/to/pos_val.txt /path/to/ner_train.iob /path/to/ner_val.iob model/
```
## Starting the webserver
```
cd src/
python3 server.py model/
```
## Building the client
The client is written in Angular and can be built using
```
cd src/client/
ng build -op ../static/
```

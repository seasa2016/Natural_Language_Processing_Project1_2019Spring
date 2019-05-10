# How to run the program
## Data Analysis and TFIDF model
1. Analysis (Analysis.ipynb)
	- To analyze the data given, we do some statistics and data visualization.
	- The ipython notebook can be run sequentially (i.e. block by block).
2. TF-IDF (TFIDF.ipynb)
	- We use VSM and cosine similarity to predict the relation of two given sentence.
	- This notebook can also be executed sequentially.

### NOTE
1. These two notebooks are developed in python3
2. Data visualization package Plotly need a account to run, that is, some configration are needed. (Detail: https://plot.ly/python/getting-started/)

## NEURAL NETWORK
please put the data under ./Text-Similarity/data/

### enviroment
please see requirements.txt

### preprocess
please go under data and run 
mkdir all_no_embedding
mkdir part_no_embedding
python all_no_embedding.py
python part_no_embedding.py


### Method list
1. SiaGRU
2. QALSTM
3. AttnGRU
4. BiMPM (Bilateral Multi-Perspective Matching for Natural Language Sentences)

### training
to train model, please run
python train.py --model ${model} --save ${PATH} --pred ${pred_type} --data ${PATH_TO_DATA} --batch_size 128 --learning_rate 0.0005

or you can run ./train_nn.sh, this will do all for you

memory should larger than 30gb, or you will crush at bimpm model XD.


### testing
to test model, please run ./test_nn.sh 
python test.py --save ${path_to_checkpoint} --data ${input_path} --out ${path_to_output}

or you can run ./test_nn.sh, this will do all for you

after run the testing, please use gen_ans.py to output the answer

### bert
for bert please run train_bert


## Feature Engineering (Feature engineering.ipynb)
#### Analysis of different feature combination

1. Original Basic Siamese RNN (LSTM)
    - Use Jieba tokenizer to get tokens
    - Build dictionary
    - Turn titles into index vectors
    - Zero padding to make fixed-length index vector
    - Turn label into one-hot vectors
    - Siamese LSTM model
    - Train and test
    - Submission

2. Pre-trained embedding
    - word2vec (character-level/ word-level)
    - doc2vec 
    - fastText (character-level/ word-level)
    - bert-as-service

3. Handcrafted features
    - TF-IDF similarity of title 1 and title 2 (refered to file ../tfidf/TF-IDF.ipynb)
    - Statistics features of rumor keywords 
    - Overlap ratio of string matching between title 1 and title 2
    - Token set ratio matching
4. Other simple models

### NOTE
1. Enviroment: python3
    - bert-serving-client==1.8.9
    - bert-serving-server==1.8.9
    - gensim==3.7.2
    - jieba==0.39
    - Keras==2.2.4
    - nltk==3.4.1
    - numpy==1.16.2
    - pandas==0.24.2
    - requests==2.21.0
    - scikit-learn==0.20.3
    - scipy==1.2.1
    - sklearn==0.0
    - tensorflow==1.13.1
    - xgboost==0.82
2. bert-as-service: https://github.com/hanxiao/bert-as-service


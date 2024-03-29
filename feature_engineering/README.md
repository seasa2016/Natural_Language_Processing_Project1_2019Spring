# Feature Engineering
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

## NOTE
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

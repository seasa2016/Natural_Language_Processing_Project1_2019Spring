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
2. bert-as-service: https://github.com/hanxiao/bert-as-service

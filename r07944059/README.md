# NEURAL NETWORK
please put the data under ./Text-Similarity/data/

## enviroment
please see requirements.txt

## preprocess
please go under data and run 
mkdir all_no_embedding
mkdir part_no_embedding
python all_no_embedding.py
python part_no_embedding.py


## Method list
1. SiaGRU
2. QALSTM
3. AttnGRU
4. BiMPM (Bilateral Multi-Perspective Matching for Natural Language Sentences)

## training
to train model, please run
python train.py --model ${model} --save ${PATH} --pred ${pred_type} --data ${PATH_TO_DATA} --batch_size 128 --learning_rate 0.0005

or you can run ./train_nn.sh, this will do all for you

memory should larger than 30gb, or you will crush at bimpm model XD.


## testing
to test model, please run ./test_nn.sh 
python test.py --save ${path_to_checkpoint} --data ${input_path} --out ${path_to_output}

or you can run ./test_nn.sh, this will do all for you

after run the testing, please use gen_ans.py to output the answer

## bert
for bert please run train_bert

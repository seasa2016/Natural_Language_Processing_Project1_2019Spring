model='attnlstm'
#word=('all_no' 'part_no')
word=('all_no' 'part_no')
#pred=('two_class' 'two_regression' 'three_class')
pred=('three_class')

for((i=0;$i<${#word[@]};i=i+1))
do
    for((j=0;$j<${#pred[@]};j=j+1))
    do
		echo ${model}_${word[i]}_${pred[j]}_8
        mkdir ./saved_models/${model}_${word[i]}_${pred[j]}_gru_8
        python train.py --model ${model} --save ${model}_${word[i]}_${pred[j]}_gru_8 --pred ${pred[j]} --data ./data/${word[i]}_embedding --batch_size 128 --learning_rate 0.001 > saved_models/${model}_${word[i]}_${pred[j]}_gru_8/log
    done
done

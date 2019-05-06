model='qalstm'
word=('all_no' 'part_no')
#word=('part_no')
pred=('two_class' 'two_regression' 'three_class')


for((i=0;$i<${#word[@]};i=i+1))
do
    for((j=0;$j<${#pred[@]};j=j+1))
    do
		echo ${model}_${word[i]}_${pred[j]}_ban
        python test.py --save ${model}_${word[i]}_${pred[j]}_ban --data ./data/${word[i]}_embedding/test.csv --out ./saved_models/${model}_${word[i]}_${pred[j]}_ban/pred
    done
done

model='siamese'
word=('all_no' 'part_no')
pred=('Two_class' 'Two_regression' 'Three_class')

for((i=0;$i<${#word[@]};i=i+1))
do
    for((j=0;$j<${#pred[@]};j=j+1))
    do
        mkdir ${model}_${word[i]}_${pred[j]}
        python train.py --model ${model} --save ${model}_${word[i]}_${pred[j]} --pred ${pred[j]} --data ./data/${word[i]}_embedding > saved_models/${model}_${word[i]}_${pred[j]}/log
    done
done
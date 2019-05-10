model=('siamese' 'qalstm' 'attnlstm' 'bimpm')
word=('all_no' 'part_no')
pred=('two_class' 'focal_two_class' 'three_class' 'two_regression')
#pred=('focal_two_class')

for((k=0;$k<${#model[@]};k=k+1))
do
    for((i=0;$i<${#word[@]};i=i+1))
    do
        for((j=0;$j<${#pred[@]};j=j+1))
        do
            echo ${model[k]}_${word[i]}_${pred[j]}
            mkdir ./Text-Similarity/saved_models/${model[k]}_${word[i]}_${pred[j]}_total
            python ./Text-Similarity/train.py --model ${model[k]} --save ${model[k]}_${word[i]}_${pred[j]}_total --pred ${pred[j]} --data ./Text-Similarity/data/${word[i]}_embedding --batch_size 128 --learning_rate 0.0005
        done
    done
done
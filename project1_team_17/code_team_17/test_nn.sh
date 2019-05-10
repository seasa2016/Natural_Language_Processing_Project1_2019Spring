model=('siamese' 'qalstm' 'attnlstm' 'bimpm')
word=('all_no' 'part_no')
pred=('two_class' 'focal_two_class' 'three_class' 'two_regression')


for((k=0;$k<${#model[@]};k=k+1))
do
    for((i=0;$i<${#word[@]};i=i+1))
    do
        for((j=0;$j<${#pred[@]};j=j+1))
        do
            echo ${model[k]}_${word[i]}_${pred[j]}
            python ./Text-Similarity/test.py --save ${model[k]}_${word[i]}_${pred[j]} --data ./Text-Similarity/data/${word[i]}_embedding/test.csv --out ./Text-Similarity/saved_models/${model[k]}_${word[i]}_${pred[j]}/pred
        done
    done
done


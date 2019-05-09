import pandas as pd
import codecs
import numpy as np
np.random.seed(5)

dictionary = {'<pad>':0,'<oov>':1}

def check(w,train=True):
    try:
        return  dictionary[w]
    except:
        if(train):
            dictionary[w] = len(dictionary)
            return  dictionary[w] 
        else:
            print(w)
            return  1

def part(sent):
    w=''
    for ch in sent:
        if( ch in 'qazwsxedcrfvtgbyhnujmikolp'):
            w += ch
        else:
            if(len(w)>0):
                yield w
                w =''
            yield ch
    if(len(w)>0):
        yield w
        
def parsing(querys,train=True):
    length = []
    new_querys = []
    if(type(querys[1])==float):
        querys[1] = querys[0]
    querys[1] = querys[1].lower()
    querys[0] = querys[0].lower()
    
    for query in querys:
        new_querys.append([])
        for w in part( query):
            new_querys[-1].append( check(w,train) )
                    
        length.append(len(new_querys[-1]))
       
    return new_querys,length

datas = {'query':[],'length':[],'label':[]}
temp = pd.read_csv('./train.csv')
for id1,id2,title1,title2,label in zip(temp['tid1'],temp['tid2'],temp['title1_zh'],temp['title2_zh'],temp['label']):
    if(id1==id2):
        continue
    elif(type(title1)==float or type(title2)==float):
        continue
    
    query,length = parsing([title1,title2])
    datas['query'].append(query)
    datas['length'].append(length)
    datas['label'].append(label)

arr = [0]*len(dictionary)
for i,w in enumerate(dictionary):
    arr[dictionary[w]] = w
with open('./all_no_embedding/vocab','w') as f:
    for w in arr:
        f.write('{0}\n'.format(w))

data = pd.DataFrame(datas)
data.reset_index().to_csv('./all_no_embedding/total.csv',index=False)
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=36)
train.reset_index().to_csv('./all_no_embedding/train.csv',index=False)
test.reset_index().to_csv('./all_no_embedding/eval.csv',index=False)

datas = {'query':[],'length':[]}
temp = pd.read_csv('./test.csv')
for title1,title2 in zip(temp['title1_zh'],temp['title2_zh']):
    query,length = parsing([title1,title2],train=False)
    datas['query'].append(query)
    datas['length'].append(length)

data = pd.DataFrame(datas)
data.to_csv('./all_no_embedding/test.csv',index=False)
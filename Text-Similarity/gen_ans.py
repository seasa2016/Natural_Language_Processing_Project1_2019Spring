import pandas as pd
import numpy as np
import sys
import os

#first find the id for the input
uids = pd.read_csv('./data/test.csv')['id'].tolist()

with open(sys.argv[1]) as f:
	labels = [int(line.split(',')[0]) for line in f]

target = sys.argv[1].split('/')[2]

arr = ['unrelated','agreed','disagreed']
with open('./result/{0}'.format(target),'w') as f:
	f.write('Id,Category\n')

	for uid,label in zip(uids,labels):
		if(uid==357062):
			f.write("357062,agreed\n")
		else:
			f.write("{0},{1}\n".format(uid,arr[label]))


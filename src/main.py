import pandas as pd
import config
import sys
import mod
import evaluate

settings = sys.argv[1:]
method = [config.methodSVM, config.methodBagging]
if '-dt' in settings:
	method[0] = config.methodDT
if '-svm' in settings:
	method[0] = config.methodSVM
if '-bagging' in settings:
	method[1] = config.methodBagging
if '-ada' in settings:
	method[1] = config.methodAdaboosting

train_df = pd.read_csv('../input/train.csv', sep='\t')
datas = train_df.loc[:, ['summary', 'reviewText']]
labels = train_df.loc[:, ['overall']]
Mod = mod.getMod(method)
Mod.train(datas)
res = Mod.test(datas)
print(len(res))
evaluate.evaluate(res, labels)

test_df = pd.read_csv('../input/test.csv', sep = '\t')
datas = test_df.loc[:, ['summary', 'reviewText']]
res = Mod.test(datas)

f = open('../output/predict.txt', 'w')
print('id,predicted', file = f)
for i in range(res.shape[0]):
	print(str(i+1)+','+str(res.loc[i].overall), file = f)

f.close()
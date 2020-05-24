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
if '-t' in settings:
	method.append(int(settings[settings.index('-t')+1]))
else:
	method.append(30)
if '-b' in settings:
	method.append(int(settings[settings.index('-b')+1]))
else:
	method.append(10000)	
train_df = pd.read_csv('../input/train.csv', sep='\t')
train_vec = pd.read_csv('../input/train_vec.csv', sep=',')
#print(train_vec)
datas = train_vec.drop('id', axis=1)
labels = pd.DataFrame(train_df.loc[:, ['overall']].values)
Mod = mod.getMod(method)
Mod.train(datas, labels)
#res = Mod.test(datas)
#print(len(res))
#evaluate.evaluate(res, labels)

test_df = pd.read_csv('../input/test_vec.csv', sep = ',')
datas = test_df.drop('id', axis=1)
res = Mod.test(datas)

f = open('../output/predict.csv', 'w')
print('id,predicted', file = f)
for i in range(res.shape[0]):
	print(str(i+1)+','+str(res.loc[i,0]), file = f)

f.close()
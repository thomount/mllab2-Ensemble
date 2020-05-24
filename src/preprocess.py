import pandas as pd
import numpy
import gensim
Len = 256

train_df = pd.read_csv('../input/train.csv', sep='\t')
datas = train_df.loc[:, ['summary', 'reviewText']]
f = open('../data/train.txt', 'w')
for i in range(datas.shape[0]):
	print(datas.loc[i].summary, datas.loc[i].reviewText, file = f)
f.close()

train_df = pd.read_csv('../input/test.csv', sep='\t')
datas = train_df.loc[:, ['summary', 'reviewText']]
f = open('../data/test.txt', 'w')
for i in range(datas.shape[0]):
	print(datas.loc[i].summary, datas.loc[i].reviewText, file = f)
f.close()


from gensim.models import word2vec
sentences=word2vec.Text8Corpus('../data/train.txt')
model=word2vec.Word2Vec(sentences, size=Len, window=5,min_count=5,workers=4)

#model = gensim.models.Word2Vec.load('../data/w2v_train_model')
#model.train(sentences)
model.save('../data/w2v_train_model')
#print(model['test'])


#model = gensim.models.Word2Vec.load('../data/w2v_train_model')
files = ['train', 'test']
for file in files:
	f = open('../data/'+file+'.txt')
	df = []
	t = 0
	for line in f.readlines():
		d = []
		t += 1
		if t % 1000 == 0:
			print(t, len(df))
		for word in line.split():
			if word.isalpha() and word in model:
				d.append(model[word])
		if len(d) > 0:
			d1 = list(numpy.mean(d, axis = 0))
		else:
			d1 = [0]*Len
		#print(len(d1))
		df.append(d1)
		#print([list(numpy.mean(d, axis = 0))])
	f.close()
	pd.DataFrame(df).to_csv('../input/'+file+'_vec.csv')


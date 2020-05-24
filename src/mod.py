import pandas as pd
import config
import random
from sklearn.svm import SVR, SVC
from sklearn import tree
import numpy
import math
class Mod:
	def __init__(self):
		pass
	def train(self, data, labels):		#用data作为数据进行训练
		pass
	def test(self, data):		#用data作为数据进行测试
		return pd.DataFrame(data=0, index=data.index)

class SVM(Mod):
	def __init__(self):
		self.linear_svr = SVC()
	def train(self, data, labels):
		print('train', data.shape)
		self.linear_svr.fit(data, labels.values.reshape(data.shape[0]))
	def test(self, data):
		print('test', data.shape)
		linear_svr_y_predict = self.linear_svr.predict(data)
		return pd.DataFrame(data=linear_svr_y_predict, index=data.index)

class DT(Mod):
	def __init__(self):
		self.clf = tree.DecisionTreeClassifier(max_depth=40)
	def train(self, data, labels):
		self.clf.fit(data, labels.values.reshape(data.shape[0]))
	def test(self, data):
		ret = self.clf.predict(data)
		return pd.DataFrame(data=ret, index=data.index)

class Sem:
	def __init__(self, N, M):
		self.n = N
		self.M = M
		self.mods = []
	def setMod(self, method):
		if method == config.methodSVM:
			self.mods = [SVM() for i in range(self.n)]
		else:
			self.mods = [DT() for i in range(self.n)]
		
	def train(self, data, labels):
		pass
	def test(self, data):

		return pd.DataFrame(data=0, index=data.index, columns=['overall'])

class Bagging(Sem):
	def __init__(self, N, M):
		super().__init__(N, M)
	def train(self, data, labels):
		for i in range(self.n):
			print('round ', i)
			set1 = random.sample(range(data.shape[0]), k=self.M)
			self.mods[i].train(data.loc[set1, :], labels.loc[set1, :])
			
	def test(self, data):
		print('start testing')
		ret = ret = self.mods[0].test(data)
		for i in range(1, self.n):
			print('round', i)
			ret = ret + self.mods[i].test(data)
		#print(ret)
		return ret / self.n


class Adaboosting(Sem):
	def __init__(self, N, M):
		super().__init__(N, M)
	def train(self, data, labels):
		#初始化权值
		weights = pd.DataFrame([1]*data.shape[0])
		#print(weights)
		ids = range(data.shape[0])
		self.w = numpy.array([0.0]*self.n)

		for i in range(self.n):
			errs = 1
			print('round', i)
#			print(weights)
			weights /= numpy.sum(weights.values)
			t = 0
			while t < 5:
				t += 1
				#set1 = ids.sample(n=self.M, replace=False,weights=weights.values.reshape(data.shape[0])).values.reshape(self.M)
				set1 = random.choices(ids, weights=weights.values, k=self.M)
				#print(set1)			
				self.mods[i].train(data.loc[set1, :], labels.loc[set1, :])
				testres = self.mods[i].test(data)
				#print(testres)
				#print((1-(testres==labels.loc[set1, :]).values*weights.loc[set1].values).reshape(500))
				errs = numpy.sum(((1-(testres==labels)).values*weights.values))
				if errs <= 0.5:
					print('error = ', errs)
					#print(((1-(testres==labels.loc[set1, :]))*((1-errs)/errs-1)+1))
					#print(weights.loc[set1])
					weights *= ((1-(testres==labels))*((1-errs)/errs-1)+1)
					#print(weights.loc[set1])
					self.w[i] = math.log((1-errs)/errs)
					print(self.w[i])
					break
			if t == 5:
				self.n = i
				break
		self.w = self.w / numpy.sum(self.w)
		print(self.n, self.w)
	def test(self, data):
		df = self.w[0]*self.mods[0].test(data)
		for i in range(1, self.n):
			print('round', i)
			df += self.w[i]*self.mods[i].test(data)
		return df

def getMod(method):
	print('setting = ', method)
	mod = None
	if method[1] == config.methodBagging:
		mod = Bagging(method[2], method[3])
	else:
		mod = Adaboosting(method[2], method[3])
	mod.setMod(method[0])
	return mod
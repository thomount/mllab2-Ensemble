import pandas as pd
class Mod:
	def __init__(self):
		pass
	def train(self, data):		#用data作为数据进行训练
		pass
	def test(self, data):		#用data作为数据进行测试
		return pd.DataFrame(data=0, index=data.index, columns=['overall'])

class SVM(Mod):
	pass

class DT(Mod):
	pass

def getMod(method):
	mod = Mod()
	return mod
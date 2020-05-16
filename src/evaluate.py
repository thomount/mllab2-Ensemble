import numpy
def evaluate(a, b):
	print('evaluating')
	if (a.shape[0] != b.shape[0]):
		print('size error')
		return
	d = (a.overall-b.overall).values.reshape(a.shape[0])
	print('loss = ', numpy.mean(d ** 2))
	return
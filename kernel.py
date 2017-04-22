import numpy as np
from sklearn.svm import SVC
from datetime import datetime

def load_file(filename='data.train'):

	file = open(filename, 'r')

	labels = []
	data = []

	for line in file:
		label_data = line.split(';')
		data_pt = label_data[1].split(',')
		labels.append(int(label_data[0]))
		data.append([float(i) for i in data_pt])

	labels = np.asarray(labels)
	data = np.asarray(data)

	return labels, data

train_labels, train_data = load_file()
test_labels, test_data = load_file('data.test')

linearSVM = SVC(kernel='linear')
rbfSVM = SVC(kernel='rbf', gamma=0.2)

startTime = datetime.now()
linearSVM.fit(train_data, train_labels)
print('Linear Kernel Runtime:', datetime.now() - startTime)

startTime = datetime.now()
rbfSVM.fit(train_data, train_labels)
print('RBF Kernel Runtime:', datetime.now() - startTime)

print('Linear Kernel Test Accuracy:', linearSVM.score(test_data, test_labels))
print('RBF Kernel Test Accuracy:', rbfSVM.score(test_data, test_labels))
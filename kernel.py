import numpy as np
from sklearn.svm import SVC

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

linearSVM.fit(train_data, train_labels)
rbfSVM.fit(train_data, train_labels)
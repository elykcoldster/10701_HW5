import numpy as np
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
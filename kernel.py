import numpy as np
from sklearn.svm import SVC
from datetime import datetime
from data import load_file

train_labels, train_data = load_file()
test_labels, test_data = load_file('data.test')

linearSVM = SVC(kernel='linear')
rbfSVM = SVC(kernel='rbf', gamma=0.2)

startTime = datetime.now()
linearSVM.fit(train_data, train_labels)
print('Linear Kernel Runtime:', (datetime.now() - startTime).total_seconds() * 1000)

startTime = datetime.now()
rbfSVM.fit(train_data, train_labels)
print('RBF Kernel Runtime:', (datetime.now() - startTime).total_seconds() * 1000)

print('Linear Kernel Test Accuracy:', linearSVM.score(test_data, test_labels))
print('RBF Kernel Test Accuracy:', rbfSVM.score(test_data, test_labels))
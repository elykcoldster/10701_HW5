import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
from sklearn import pipeline
from datetime import datetime
from data import load_file

train_labels, train_data = load_file()
test_labels, test_data = load_file('data.test')

samples = [50, 100, 150, 200, 250, 300, 350, 400, 450]

training_times = []
test_accuracies = []

linearSVM = SVC(kernel='linear')
feature_map_fourier = RBFSampler(gamma=0.2, random_state=1)
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", linearSVM)])

for n in samples:	
	fourier_approx_svm.set_params(feature_map__n_components=n)
	startTime = datetime.now()
	fourier_approx_svm.fit(train_data, train_labels)
	training_times.append((datetime.now() - startTime).total_seconds()*1000)

	test_accuracies.append(fourier_approx_svm.score(test_data, test_labels))
	
print(training_times)
print(test_accuracies)

plt.title('Test Accuracy vs. Feature Dimensions')
plt.xlabel('Feature Dimensions')
plt.ylabel('Test Accuracy')
plt.plot(samples, test_accuracies)
plt.show()

plt.title('Training Time vs. Feature Dimensions')
plt.xlabel('Feature Dimensions')
plt.ylabel('Training Time (ms)')
plt.plot(samples, training_times)
plt.show()
from collections import Counter
import cv2
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import sys
import time
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

def print_same_line(string):
	sys.stdout.write('\r' + string)
	sys.stdout.flush()

"""
CIFAR-10 Dataset: "Learning Multiple Layers of Features from Tiny Images" Alex Krizhevsky, 2009.
"""
class CIFAR10:
	def __init__(self, data_path):
		"""Extracts CIFAR10 data from data_path"""
		file_names = ['data_batch_%d' % i for i in range(1,6)]
		file_names.append('test_batch')

		X = []
		y = []
		for file_name in file_names:
			with open(data_path + file_name) as fin:
				data_dict = cPickle.load(fin)
			X.append(data_dict['data'].ravel())
			y = y + data_dict['labels']

		self.X = np.asarray(X).reshape(60000, 32*32*3)
		self.y = np.asarray(y)

		fin = open(data_path + 'batches.meta')
		self.LABEL_NAMES = cPickle.load(fin)['label_names']
		fin.close()

	def train_test_split(self):
		X_train = self.X[:50000]
		y_train = self.y[:50000]
		X_test = self.X[50000:]
		y_test = self.y[50000:]

		return X_train, y_train, X_test, y_test

	def all_data(self):
		return self.X, self.y

	def __prep_img(self, idx):
		img = self.X[idx].reshape(3,32,32).transpose(1,2,0).astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return img

	def show_img(self, idx):
		cv2.imshow(self.LABEL_NAMES[self.y[idx]], self.__prep_img(idx))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def show_examples(self):
		fig, axes = plt.subplots(5, 5)
		fig.tight_layout()
		for i in range(5):
			for j in range(5):
				rand = np.random.choice(range(self.X.shape[0]))
				axes[i][j].set_axis_off()
				axes[i][j].imshow(self.__prep_img(rand))
				axes[i][j].set_title(self.LABEL_NAMES[self.y[rand]])
		plt.show()
		
class NearestNeighbour:
    def __init__(self, distance_func='l1'):
        self.distance_func = distance_func
    
    def train(self, X, y):
        """ X NxD matrix s.t. each row is a training example """
        """ y Nx1 matrix of true values """
        self.X_tr = X.astype(np.float32) #prevents issues with unsigned values
        self.y_tr = y
    
    def predict(self, X):
        """ X MxD matrix s.t. each row is a testing example """
        X_te = X.astype(np.float32)
        num_test_examples = X.shape[0]
        y_pred = np.zeros(num_test_examples, self.y_tr.dtype)
        
        for i in range(num_test_examples):
            if self.distance_func == 'l2':
                distances = np.sum(np.square(self.X_tr - X_te[i]), axis=1) #dropped sqrt as monotonic
            else:
                """ X_tr NxD matrix, X_te[i] is a 1xD matrix, its one row """
                """ numpy allows subtraction override (due to matching D dim) """ 
                """ so X_te[i] is subtracted from each row in X_tr """
                distances = np.sum(np.abs(self.X_tr - X_te[i]), axis=1) #gives one column
                
            smallest_dist_idx = np.argmin(distances)
            """ transfers label from closest img index """
            y_pred[i] = self.y_tr[smallest_dist_idx]
        return y_pred

set = CIFAR10('./cifar-10-batches-py/')
X_train, y_train, X_test, y_test = dataset.train_test_split()
X, y = dataset.all_data()

dataset.show_examples()

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

knn = KNeighborsClassifier(n_neighbors=5, p=1, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = np.mean(y_test == y_pred)
print accuracy

param_grid = {'n_neighbors': [1, 3, 5, 10, 20, 50, 100], 'p': [1, 2]}
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print grid_search.best_params_



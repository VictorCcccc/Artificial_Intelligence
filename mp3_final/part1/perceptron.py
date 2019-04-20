import numpy as np

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model. 

		This function will initialize a feature_dim weight vector,
		for each class. 

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS] 
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		"""

		self.w = np.zeros((feature_dim+1,num_class))
		self.num_class = num_class

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		for i in range(5):
			for examples in range(len(train_label)):
				feature = train_set[examples,:]
				feature = np.append(feature,1)
				values = np.zeros(self.num_class)
				for label in range(self.num_class):
					values[label] = np.dot(feature,self.w[:,label])

				prediction = np.argmax(values)
				realistic = train_label[examples]
				if prediction != realistic:
					eta = 1/(examples + 1)
					self.w[:, prediction] -= eta * feature
					self.w[:, realistic] += eta * feature

		pass

	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset. 
			The accuracy is computed as the average of correctness 
			by comparing between predicted label and true label. 
			
		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value 
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""    

		# YOUR CODE HERE
		accuracy = 0

		class_selection = []
		item_list = []
		for i in range(10):
			class_selection.append([i,0])
			item_list.append([i,0])
		pred_label = np.zeros((len(test_set)))
		pred = []

		for examples in range(len(test_label)):
			feature = test_set[examples,:]
			feature = np.append(feature,1)
			values = np.zeros(self.num_class)
			for label in range(self.num_class):
				values[label] = np.dot(feature,self.w[:,label])
			pred_label[examples] = np.argmax(values)
			pred.append(np.max(values))
		accuracy = 1 - np.count_nonzero(pred_label - test_label) / len(test_label)
        
        
		# for examples in range(len(test_label)):
		# 	index = pred[examples][0]
		# 	value = pred[examples][1]
		# 	if value > class_selection[index][1]:
		# 		class_selection[index][1] = value
		# 		item_list[index][1] = examples


		# ac = np.zeros(self.num_class)
		# count = np.zeros(self.num_class)
		# for i in range(len(test_label)):
		# 	if test_label[i] == pred_label[i]:
		# 		ac[test_label[i]]+=1
		# 	count[test_label[i]]+=1
		# for i in range(self.num_class):
		# 	print(i,ac[i]/count[i])

		pass
		
		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters 
		""" 

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters 
		""" 

		self.w = np.load(weight_file)


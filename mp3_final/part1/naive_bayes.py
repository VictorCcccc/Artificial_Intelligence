import numpy as np

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model. 

		This function will initialize prior and likelihood, where 
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of 
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		    num_value(int): number of possible values for each pixel 
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim
		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset. 
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
				(# of features/pixels per image, # of possible values per pixel, # of class).
			You should apply Laplace smoothing to compute the likelihood. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""
		# YOUR CODE HERE
		k = 0.1
		for label in train_label:
			self.prior[label] += 1 / len(train_label)
		count = np.zeros((self.feature_dim, self.num_value, self.num_class))
		for examples in range(len(train_set)):
			for pixel in range(len(train_set[0])):
				label = train_label[examples]
				value = train_set[examples,pixel]
				p_prior = self.prior[label]
				count[pixel,value,label] += 1
		for label in range(self.num_class):
			laplace = np.ones((self.feature_dim, self.num_value))* k
			self.likelihood[:,:,label] = (count[:,:,label]+laplace)/(len(train_label)*p_prior + k * self.num_value)


		pass

	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.  
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
		pred_label = np.zeros((len(test_set)))

		pred_value = np.zeros((len(test_set),self.num_class))
		for label in range(self.num_class):
			pred_value[:,label] = np.log(self.prior[label])
			for examples in range(len(test_set)):
				for pixel in range(len(test_set[0])):
					value = test_set[examples,pixel]
					pred_value[examples,label] += np.log(self.likelihood[pixel,value,label])
					
		pred_label = np.argmax(pred_value, axis=1)
		accuracy = 1 - np.count_nonzero(pred_label - test_label) / len(test_label)

        
		# class_selection = []
		# item_list = []
		# for i in range(10):
		# 	class_selection.append([i, 0])
		# 	item_list.append([i, 0])
		# label_value = np.max(pred_value,axis=1)
		# for example in range(len(test_set)):
		# 	index = pred_label[example]
		# 	value = label_value[example]
		# 	if value < class_selection[index][1]:
		# 		class_selection[index][1] = value
		# 		item_list[index][1] = example


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

	def save_model(self, prior, likelihood):
		""" Save the trained model parameters 
		"""    

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters 
		""" 

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
		"""
		Get the feature likelihoods for high intensity pixels for each of the classes,
		    by sum the probabilities of the top 128 intensities at each pixel location,
		    sum k<-128:255 P(F_i = k | c).
		    This helps generate visualization of trained likelihood images.

		Args:
		    likelihood(numpy.ndarray): likelihood (in log) with a dimension of
		        (# of features/pixels per image, # of possible values per pixel, # of class)
		Returns:
		    feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
		        (# of features/pixels per image, # of class)
		"""

		feature_likelihoods = np.zeros((likelihood.shape[0], likelihood.shape[2]))
		for label in range(self.num_class):
			pixel_value = np.argsort(likelihood[:,:,label],axis = 1)
			for pixel in range(self.feature_dim):
				for i in range(128,256):
					index = pixel_value[pixel,i]
					value = likelihood[pixel,index,label]
					feature_likelihoods[pixel,label] += value
                

		return feature_likelihoods

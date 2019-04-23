import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.utils.multiclass import unique_labels

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    losses = []
    num_examples = len(y_train)
    
    #IMPLEMENT HERE
    for e in range(epoch):
        loss = 0
        accuracy = 0
        batch_size = int(num_examples/200)
        
		# Shuffle
        if shuffle:
            dataset = np.column_stack((x_train, y_train))
            np.random.shuffle(dataset)
            row = np.shape(dataset)[1]
            x_train = dataset[:,:row-1]
            y_train = dataset[:,row-1]
            
        for i in range(batch_size):
            x_train_batch = x_train[200*i:200*(i+1),:]
            y_train_batch = y_train[200*i:200*(i+1)]
            
            w1, w2, w3, w4, b1, b2, b3, b4, loss_, pred = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_train_batch, y_train_batch, num_classes)
            accuracy_ = (num_examples-np.count_nonzero(pred-y_train_batch))/num_examples
            loss += loss_
            accuracy += accuracy_/batch_size
        losses.append(loss)
           
    return w1, w2, w3, w4, b1, b2, b3, b4, losses


"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""

def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    w1, w2, w3, w4, b1, b2, b3, b4, loss, pred = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes)
    num_examples = len(y_test)
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes

    correct_per_class = [0]*num_classes
    total_per_class = [0]*num_classes
    
    avg_class_rate = (num_examples-np.count_nonzero(pred-y_test))/num_examples

    for i in range(len(pred)):
        label = int(y_test[i])
        if int(pred[i]) == label:
            correct_per_class[label] += 1
        total_per_class[label] += 1

    for i in range(num_classes):
        class_rate_per_class[i] = correct_per_class[i]/total_per_class[i]
        
#    class_names = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress",
#                            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
#    
#    plot_confusion_matrix(y_test, pred, classes=class_names, normalize=True,
#                      title='Confusion matrix, with normalization')

    return avg_class_rate, class_rate_per_class




"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    
    eta = 0.1
    
    (Z1,acache_1) = affine_forward(x_test,w1,b1)
    (A1,rcache_1) = relu_forward(Z1)
    (Z2,acache_2) = affine_forward(A1,w2,b2)
    (A2,rcache_2) = relu_forward(Z2)
    (Z3,acache_3) = affine_forward(A2,w3,b3)
    (A3,rcache_3) = relu_forward(Z3)
    (F,acache_4) = affine_forward(A3,w4,b4)
    
    pred = np.argmax(np.exp(F),axis = 1)

    
    (loss,dF) = cross_entropy(F, y_test)
    (dA3,dw4,db4) = affine_backward(dF, acache_4)
    dZ3 = relu_backward(dA3,rcache_3)
    (dA2,dw3,db3) = affine_backward(dZ3, acache_3)
    dZ2 = relu_backward(dA2,rcache_2)    
    (dA1,dw2,db2) = affine_backward(dZ2, acache_2)
    dZ1 = relu_backward(dA1,rcache_1)
    (dX,dw1,db1) = affine_backward(dZ1, acache_1)
    
    w1 -= eta*dw1
    w2 -= eta*dw2
    w3 -= eta*dw3
    w4 -= eta*dw4
    
    b1 -= eta*db1
    b2 -= eta*db2
    b3 -= eta*db3
    b4 -= eta*db4
    
    
    return w1, w2, w3, w4, b1, b2, b3, b4, loss, pred

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    
    Z = np.dot(A,W) + b
    cache = (A,W,b)
    
    return Z, cache

def affine_backward(dZ, cache):
    dA = np.dot(dZ,cache[1].T)
    dW = np.dot(cache[0].T, dZ)
    dB = np.sum(dZ,axis = 0)
    return dA, dW, dB

def relu_forward(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    dZ = np.array(dA, copy = True)
    dZ[cache <= 0] = 0;
    return dZ

def cross_entropy(F, y):
    
    num_examples = F.shape[0]
    exp_F = np.exp(F)
    p = exp_F / np.sum(exp_F, axis=1, keepdims=True)
    y = y.astype(np.int)
    correction = -np.log(p[range(num_examples),y])
    loss = np.sum(correction)/num_examples
    p[range(num_examples),y] -= 1
    dF = p / num_examples
    return loss, dF

## Confusion Matrix

#def plot_confusion_matrix(y_true, y_pred, classes,
#                          normalize=False,
#                          title=None,
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if not title:
#        if normalize:
#            title = 'Normalized confusion matrix'
#        else:
#            title = 'Confusion matrix, without normalization'
#
#    # Compute confusion matrix
#    cm = confusion_matrix(y_true, y_pred)
#    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
##    print(cm)
#
#    fig, ax = plt.subplots()
#    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
#    # We want to show all ticks...
#    ax.set(xticks=np.arange(cm.shape[1]),
#           yticks=np.arange(cm.shape[0]),
#           # ... and label them with the respective list entries
#           xticklabels=classes, yticklabels=classes,
#           title=title,
#           ylabel='True label',
#           xlabel='Predicted label')
#
#    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")
#
#    # Loop over data dimensions and create text annotations.
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i in range(cm.shape[0]):
#        for j in range(cm.shape[1]):
#            ax.text(j, i, format(cm[i, j], fmt),
#                    ha="center", va="center",
#                    color="white" if cm[i, j] > thresh else "black")
#    fig.tight_layout()
#    plt.plot()
#    return ax



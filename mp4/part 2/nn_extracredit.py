# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:40:25 2019

@author: Victor
"""
# Modified according to the kears documentation https://keras.io/getting-started/sequential-model-guide/


from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

x_train = np.load("data/x_train.npy")
train_data = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
train_data = train_data.reshape(50000,28,28,1)
train_labels = np.load("data/y_train.npy")
train_labels = train_labels.astype(np.int)

x_test = np.load("data/x_test.npy")
eval_data = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
eval_data = eval_data.reshape(10000,28,28,1)
eval_labels = np.load("data/y_test.npy")
eval_labels = eval_labels.astype(np.int)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(class_names)

#create model
model = keras.Sequential()
#add model layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, validation_data=(eval_data, eval_labels), epochs=20)

test_loss, test_acc = model.evaluate(eval_data, eval_labels)

print('Test accuracy:', test_acc)

pred_labels = np.argmax(model.predict(eval_data),axis = 1)

#Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.plot()
    return ax



plot_confusion_matrix(pred_labels, eval_labels, classes=class_names, normalize=True,
                  title='Confusion matrix, with normalization')
plt.show()

class_rate_per_class = [0.0] * num_classes
correct_per_class = [0]*num_classes
total_per_class = [0]*num_classes
for i in range(len(pred_labels)):
    label = int(eval_labels[i])
    if int(pred_labels[i]) == label:
        correct_per_class[label] += 1
    total_per_class[label] += 1

for i in range(num_classes):
    class_rate_per_class[i] = correct_per_class[i]/total_per_class[i]
    
print('class rate', class_rate_per_class)

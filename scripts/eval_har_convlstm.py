#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a pretrained model saved as *.h5 using 'testData_X.npy'
and 'groundTruth_X.npy'. Error reported is the cross entropy loss in percentage. Also generates a png file for the confusion matrix.
Based on work by Muhammad Shahnawaz.
"""

import matplotlib
matplotlib.use('Agg')

# importing the dependencies
from keras.models import load_model, Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm

## define a function for plotting the confusion matrix
## takes cmNormalized
os.environ['QT_PLUGIN_PATH'] = ''
def plot_cm(cM, labels,title):

    plt.close()
    ## normalizing the confusionMatrix for showing the probabilities
    cmNormalized = np.around((cM/cM.astype(np.float).sum(axis=1)[:,None])*100,2)
    ## creating a figure object
    fig = plt.figure()
    ## plotting the confusion matrix
    plt.imshow(cmNormalized,interpolation='bilinear',cmap = plt.cm.Purples)
    ## creating a color bar and setting the limits
    plt.colorbar()
    plt.clim(0,100)
    ## assiging the title, x and y labels
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth')
    plt.title(title + '\n%age confidence')
    ## defining the ticks for the x and y axis
    plt.xticks(range(len(labels)),labels,rotation = 60)
    plt.yticks(range(len(labels)),labels)
    ## number of occurences in the boxes
    width, height = cM.shape
    print('Accuracy for each class is given below.')
    for predicted in range(width):
        for real in range(height):
            color = 'black'
            if(predicted == real):
                color = 'white'
                print(labels[predicted].ljust(12)+ ':', cmNormalized[predicted,real], '%')
            plt.gca().annotate(
                    '{:d}'.format(int(cmNormalized[predicted,real])),xy=(real, predicted),
                    horizontalalignment = 'center',verticalalignment = 'center',color = color)

    ## making sure that the figure is not clipped
    plt.tight_layout()
    plt.grid('off')
#     ## turn off white grid lines
#     plt.grid(False)
#     ax.grid(False)
    ## saving the figure
    fig.savefig(title +'.png')

## loading the pretrained model
model = load_model('./data/model_had_lstm_logo.h5')

## load weights into new model
model.load_weights("./data/model_weights_had_lstm_logo.h5")
print("Loaded model from disk")

## loading the testData and groundTruth data
test_x = np.load('./data/testData_had_lstm_logo.npy')
groundTruth = np.load('./data/groundTruth_had_lstm_logo.npy')

## evaluate loaded model on test data
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(test_x,groundTruth,verbose=2)

## print out values for metrics
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print('Baseline Error: %.2f%%' %(100-score[1]*100))

## Creating and plotting a confusion matrix

## defining the 12 class labels
labels = ['WalkForward','WalkLeft','WalkRight','WalkUp','WalkDown','RunForward', 'JumpUp', 'Sit', 'Stand', 'Sleep', 'ElevatorUp', 'ElevatorDown']

## predicting the classes
predictions = model.predict(test_x,verbose=2)

## getting the class predicted and class in ground truth for creation of confusion matrix
predictedClass = np.zeros((predictions.shape[0]))
groundTruthClass = np.zeros((groundTruth.shape[0]))

for instance in range (groundTruth.shape[0]):
    predictedClass[instance] = np.argmax(predictions[instance,:])
    groundTruthClass[instance] = np.argmax(groundTruth[instance,:])

cm = metrics.confusion_matrix(groundTruthClass,predictedClass)

print(cm)

## plotting the confusion matrix
plot_cm(cm, labels,'./plots/confusion_matrix_90_logo')

print(model.summary())

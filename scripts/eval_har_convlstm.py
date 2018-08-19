#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:07:51 2017
This script is written to evaluate a pretrained model saved as  model.h5 using 'testData.npy'
and 'groundTruth.npy'. This script reports the error as the cross entropy loss in percentage
and also generated a png file for the confusion matrix.
Based on work by Muhammad Shahnawaz
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

# defining a function for plotting the confusion matrix
# takes cmNormalized
os.environ['QT_PLUGIN_PATH'] = ''
def plot_cm(cM, labels,title):
    # normalizing the confusionMatrix for showing the probabilities
    # cmNormalized = np.around((cM/cM.sum(axis=1)[:,None])*100,2)
    cmNormalized = np.around((cM/cM.astype(np.float).sum(axis=1)[:,None])*100,2)

    # C / C.astype(np.float).sum(axis=1)


    # print cM
    # print cmNormalized
    # creating a figure object
    fig = plt.figure()
    # plotting the confusion matrix
    plt.imshow(cmNormalized,interpolation=None,cmap = plt.cm.Blues)
    # creating a color bar and setting the limits
    plt.colorbar()
    plt.clim(0,100)
    # assiging the title, x and y labels
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth')
    plt.title(title + '\n%age confidence')
    # defining the ticks for the x and y axis
    plt.xticks(range(len(labels)),labels,rotation = 60)
    plt.yticks(range(len(labels)),labels)
    # number of occurences in the boxes
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
    # making sure that the figure is not clipped
    plt.tight_layout()
    # saving the figure
    fig.savefig(title +'.png')

# loading the pretrained model
model = load_model('model_had_lstm_1out.h5')

# load weights into new model
model.load_weights("model_weights_had_lstm_1out.h5")
print("Loaded model from disk")




#loading the testData and groundTruth data
test_x = np.load('testData_had_lstm_1out.npy')
groundTruth = np.load('groundTruth_had_lstm_1out.npy')

# # evaluating the model
# score = model.evaluate(test_x,groundTruth,verbose=2)
# print('Baseline Error: %.2f%%' %(100-score[1]*100))


# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(test_x,groundTruth,verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print('Baseline Error: %.2f%%' %(100-score[1]*100))



# 1. Walking Forward
# 2. Walking Left
# 3. Walking Right
# 4. Walking Upstairs
# 5. Walking Downstairs
# 6. Running Forward
# 7. Jumping Up
# 8. Sitting
# 9. Standing
# 10. Sleeping
# 11. Elevator Up
# 12. Elevator Down

'''
 Creating and plotting a confusion matrix

'''
# defining the class labels
labels = ['WalkForward','WalkLeft','WalkRight','WalkUp','WalkDown','RunForward', 'JumpUp', 'Sit', 'Stand', 'Sleep', 'ElevatorUp', 'ElevatorDown']

# predicting the classes
predictions = model.predict(test_x,verbose=2)

print predictions
print labels

# getting the class predicted and class in ground truth for creation of confusion matrix
predictedClass = np.zeros((predictions.shape[0]))
groundTruthClass = np.zeros((groundTruth.shape[0]))
for instance in range (groundTruth.shape[0]):
    predictedClass[instance] = np.argmax(predictions[instance,:])
    groundTruthClass[instance] = np.argmax(groundTruth[instance,:])

# obtaining a confusion matrix
cm = metrics.confusion_matrix(groundTruthClass,predictedClass)

print cm
# plotting the confusion matrix
plot_cm(cm, labels,'confusion_matrix_convlstm')




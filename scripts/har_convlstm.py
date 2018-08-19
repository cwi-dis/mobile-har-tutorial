# coding: utf-8

# importing libraries and dependecies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, LSTMCell, Bidirectional, TimeDistributed, InputLayer, ConvLSTM2D
from sklearn.model_selection import LeaveOneGroupOut
import dill
#from keras import backend as K
from keras import optimizers
#K.set_image_dim_ordering('th')

import os
import scipy.io
import pickle
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import tensorflow as tf
import seaborn as sns
import pylab
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score

#print(device_lib.list_local_devices())

# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# matplotlib inline
plt.style.use('ggplot')

# defining function for loading the dataset
def readData(filePath):
    # attributes of the dataset
    columnNames = ['user_id','activity','timestamp','x-axis','y-axis','z-axis']
    data = pd.read_csv(filePath,header = None, names=columnNames,na_values=';')
    return data[0:2000]
# defining a function for feature normalization
# (feature - mean)/stdiv

def featureNormalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset-mu)/sigma

# defining the function to plot a single axis data
def plotAxis(axis,x,y,title):
    axis.plot(x,y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y)-np.std(y),max(y)+np.std(y)])
    axis.set_xlim([min(x),max(x)])
    axis.grid(True)

# defining a function to plot the data for a given activity
def plotActivity(activity,data):
    fig,(ax0,ax1,ax2) = plt.subplots(nrows=3, figsize=(15,10),sharex=True)
    plotAxis(ax0,data['timestamp'],data['x-axis'],'x-axis')
    plotAxis(ax1,data['timestamp'],data['y-axis'],'y-axis')
    plotAxis(ax2,data['timestamp'],data['z-axis'],'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()

# defining a window function for segmentation purposes
def windows(data,size):
    start = 0
    while start< data.count():
        yield int(start), int(start + size)
        start+= (size/2)

## our segmenrtation function to get streams of 90 samples in each timestep
def segment_signal_ucd(data, window_size = 90):
    segments = np.empty((0,window_size,6))
    labels= np.empty((0))
    subjects = np.empty((0))
#     print subjects

    for (start, end) in windows(data['activity'],window_size):
        x = data['acc_x'][start:end]
        y = data['acc_y'][start:end]
        z = data['acc_z'][start:end]
        p = data['gyr_x'][start:end]
        q = data['gyr_y'][start:end]
        r = data['gyr_z'][start:end]


        if(len(data['activity'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x,y,z,p,q,r])])
            if labels is not None:
                labels = np.append(labels,stats.mode(data['activity'][start:end])[0][0])
            subjects = np.append(subjects,stats.mode(data['subject'][start:end])[0][0])
    return segments, labels, subjects


## read in the USC-HAD data
DIR = './data/USC-HAD/data/'

# activity = []
subject = []
# age = []
act_num = []
sensor_readings = []

def read_dir(directory):
    for path, subdirs, files in os.walk(DIR):
        for name in files:
            if name.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(path, name))
#                 activity.append(mat['activity'])
                subject.extend(mat['subject'])
#                 age.extend(mat['age'])
                sensor_readings.append(mat['sensor_readings'])

                if mat.get('activity_number') is None:
                    act_num.append('11')
                else:
                    act_num.append(mat['activity_number'])
    return subject, act_num, sensor_readings

## Corrupt datapoint
# act_num[258] = '11'

subject, act_num, sensor_readings = read_dir(DIR)

## Get acc + gyr sensor readings and put in df (dataframe)
acc_x = []
acc_y = []
acc_z = []
gyr_x = []
gyr_y = []
gyr_z = []

act_label = []
subject_id = []
df = None

for i in range(840):
    for j in sensor_readings[i]:

        acc_x.append(j[0]) # acc_x
        acc_y.append(j[1]) # acc_y
        acc_z.append(j[2]) # acc_z
        gyr_x.append(j[3]) # gyr_x
        gyr_y.append(j[4]) # gyr_y
        gyr_z.append(j[5]) # gyr_z
        act_label.append(act_num[i])
        subject_id.append(subject[i])

df = pd.DataFrame({'subject':subject_id,'acc_x':acc_x,'acc_y':acc_y,'acc_z':acc_z,'gyr_x':gyr_x,'gyr_y':gyr_y,'gyr_z':gyr_z,'activity':act_label})


df = df[['subject','acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z','activity']]

df.loc[df['activity'] == '1', 'activity'] = 'Walking Forward'
df.loc[df['activity'] == '2', 'activity'] = 'Walking Left'
df.loc[df['activity'] == '3', 'activity'] = 'Walking Right'
df.loc[df['activity'] == '4', 'activity'] = 'Walking Upstairs'
df.loc[df['activity'] == '5', 'activity'] = 'Walking Downstairs'
df.loc[df['activity'] == '6', 'activity'] = 'Running Forward'
df.loc[df['activity'] == '7', 'activity'] = 'Jumping Up'
df.loc[df['activity'] == '8', 'activity'] = 'Sitting'
df.loc[df['activity'] == '9', 'activity'] = 'Standing'
df.loc[df['activity'] == '10', 'activity'] = 'Sleeping'
df.loc[df['activity'] == '11', 'activity'] = 'Elevator Up'
df.loc[df['activity'] == '12', 'activity'] = 'Elevator Down'

## These are the 12 classes we want to recognize!
df['activity'].unique()

## print size of dataset
print 'df size ' + str(len(df))

# segmenting the signal in overlapping windows of 90 samples with 50% overlap

segments, labels, subjects = segment_signal_ucd(df)

# store all in pickle dumps
pickle.dump(segments, open( "segments_90_logo.p","wb"))
pickle.dump(labels, open( "labels_90_logo.p","wb"))
pickle.dump(subjects, open( "subjects_90_logo.p","wb"))

groups = np.array(subjects)
logo = LeaveOneGroupOut()
logo.get_n_splits(segments, labels, groups)
# logo.get_n_splits(groups=groups)

# defining parameters for the input and network layers
# we are treating each segmeent or chunk as a 2D image (90 X 3)
numOfRows = segments.shape[1]
numOfColumns = segments.shape[2]

# reshaping the data for network input
reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
# categorically defining the classes of the activities
labels = np.asarray(pd.get_dummies(labels),dtype = np.int8)

# # open a file, where you stored the pickled data
# segments = open('segments_90_logo.p','rb')
# labels = open('labels_90_logo.p', 'rb')
# subjects = open('subjects_90_logo.p', 'rb')

# # dump information to that file
# segments = pickle.load(segments)
# labels = pickle.load(labels)

# subjects = pickle.load(subjects)


# ==================================================================================

# splitting in training and testing data
# trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
# trainX = reshapedSegments
# testX = reshapedSegments_test
# trainX = np.nan_to_num(trainX)
# testX = np.nan_to_num(testX)
# trainY = labels
# testY = labels_test

# print "segments shape:" + str(segments.shape)
# print "labels shape:" + str(labels.shape)
# print "trainX shape: " + str(trainX.shape)
# print "trainY shape: " + str(trainY.shape)
# print "testX shape: " + str(testX.shape)
# print "testY shape: " + str(testY.shape)


## Hyperparameters

numChannels = 1
numFilters = 128 # number of filters in Conv2D layer
# kernal size of the Conv2D layer
kernalSize1 = 2
# max pooling window size
poolingWindowSz = 2
# number of filters in fully connected layers
numNueronsFCL1 = 128
numNueronsFCL2 = 128
# split ratio for test and validation
# trainSplitRatio = 0.8
# number of epochs
Epochs = 20
# batchsize
batchSize = 10
# number of total clases
numClasses = labels.shape[1]
# dropout ratio for dropout layer
dropOutRatio = 0.2

## Our ConvLSTM model

def Conv2D_LSTM_Model():
    model = Sequential()

    # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
    model.add(ConvLSTM2D(numFilters, (kernalSize1,kernalSize1),input_shape=(None, numOfRows, numOfColumns, 1),activation='relu', padding='same',return_sequences=True))
    print (model.output_shape)

    # adding a maxpooling layer
    model.add(TimeDistributed(MaxPooling2D(pool_size=(poolingWindowSz,poolingWindowSz),padding='valid')))
    print (model.output_shape)

    # adding a dropout layer for the regularization and avoiding over fitting
    model.add(Dropout(dropOutRatio))
    print (model.output_shape)
#     model.add(LSTM(32,input_shape=(numOfRows, numOfColumns, 1),return_sequences=True))

    # flattening the output in order to apply the fully connected layer
    model.add(TimeDistributed(Flatten()))
    print (model.output_shape)
    # adding first fully connected layer with 256 outputs
    model.add(Dense(numNueronsFCL1, activation='relu'))
    print (model.output_shape)

    #adding second fully connected layer 128 outputs
    model.add(Dense(numNueronsFCL2, activation='relu'))
    print (model.output_shape)

    model.add(TimeDistributed(Flatten()))
    print (model.output_shape)

    # adding softmax layer for the classification
    model.add(Dense(numClasses, activation='softmax'))

    # Compiling the model to generate a model
    adam = optimizers.Adam(lr = 0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

## Train the network!
tf.get_default_graph()

cvscores = []



for index, (train_index, test_index) in enumerate(logo.split(reshapedSegments, labels, groups)):

    print "Training on fold " + str(index+1) + "/14..."

    # print("TRAIN:", train_index, "TEST:", test_index)
    trainX, testX = reshapedSegments[train_index], reshapedSegments[test_index]
    trainY, testY = labels[train_index], labels[test_index]
#     print(np.nan_to_num(trainX), np.nan_to_num(testX), trainY, testY)

    # clear model, and create it
    model = None
    model = Conv2D_LSTM_Model()

    for layer in model.layers:
        print(layer.name)
    print trainX.shape

    history = model.fit(np.expand_dims(trainX,1),np.expand_dims(trainY,1), validation_data=validation_data=(testX,testY), epochs=Epochs,batch_size=batchSize,verbose=2)
    # dill.dump(history, open( "model_" + str(index) + "_history.p","wb"))

    score = model.evaluate(np.expand_dims(testX,1),np.expand_dims(testY,1),verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    print('Baseline ConvLSTM Error: %.2f%%' %(100-score[1]*100))
    cvscores.append(score[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print(history.history.keys())
##  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig('./plots/acc_plot_logo.pdf', bbox_inches='tight')

## "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig('./plots/loss_plot_logo.pdf', bbox_inches='tight')


## Save your model!
model.save('model_had_lstm_logo.h5')
model.save_weights('model_weights_had_lstm_logo.h5')
np.save('groundTruth_had_lstm_logo.npy',np.expand_dims(testY,1))
np.save('testData_had_lstm_logo.npy',np.expand_dims(testX,1))


## write to JSON, in case you wanrt to work with that data format later when inspecting your model
with open("./data/model_hcd_test.json", "w") as json_file:
    json_file.write(model.to_json())

## write cvscores to file
with open('cvscores_convlstm_logo.txt', 'w') as cvs_file:
    cvs_file.write("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

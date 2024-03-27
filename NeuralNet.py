from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from math import log2
import random
import pandas as pd
from datagen import printProgressBar

BATCH_SIZE = 32

MODELNAME = "ResnetClassifier"
CKPT = os.path.dirname(os.path.realpath(__file__)) + "\\" + MODELNAME + "\\model"

try:
    os.makedirs(CKPT)
except OSError as error:
    print()

inj_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99)
death_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99)

def makeCls():

    #Function to create a residual block
    def resBlock(inp, cha):
        out = Dense(cha)(inp)
        out = tf.keras.activations.sigmoid(out)
        out = Dense(cha)(out)
        out += inp
        return tf.keras.activations.sigmoid(out)

    #generates the classifier network.

    #initializer Before residual Block
    inZ = Input((200), name = 'Input Vector')
    pre = Dense(512)(inZ)
    pre = LeakyReLU(0.2)(pre)

    z = resBlock(pre, 512)
    z = resBlock(z, 512)
    z = Dense(256, activation='sigmoid')(z)

    injury = resBlock(z, 256)
    # death = resBlock(z, 256)
    
    injury = Dense(1, name="injury_pred", activation='sigmoid')(injury)
    # death = Dense(1, name="fatal_pred", activation='sigmoid')(injury)

    model = tf.keras.Model(inputs=inZ, outputs=[injury])
    return model

inj_cls = makeCls()
death_cls = makeCls()

@tf.function
def trainStep(features, labels, death_features = None, death_labels = None):

    with tf.GradientTape() as inj_clsTape, tf.GradientTape() as death_clsTape:

        predictions = [tf.squeeze(inj_cls(features, training=True)), tf.squeeze(death_cls(features, training=True))]
        

        cls_loss_func = tf.keras.losses.BinaryCrossentropy()

        inj_cls_loss = cls_loss_func(labels[0], predictions[0])
        death_cls_loss = cls_loss_func(labels[1], predictions[1])

        #Apply a higher penalty for getting true death labels incorrect.
        if(death_features != None):
            death_predictions = death_cls(death_features, training=True)
            death_cls_loss += cls_loss_func(death_labels[1], death_predictions) * 5
        

    inj_cls_gradients = inj_clsTape.gradient(inj_cls_loss,inj_cls.trainable_variables)
    death_cls_gradients = death_clsTape.gradient(death_cls_loss,death_cls.trainable_variables)

    inj_opt.apply_gradients(zip(inj_cls_gradients,inj_cls.trainable_variables))
    death_opt.apply_gradients(zip(death_cls_gradients,death_cls.trainable_variables))
    return 

#Conduct a validation step to ensure model health.
def accuracy(data, labels):
    sample_num = 0
    injury_tp = 0
    injury_tn = 0
    injury_fp = 0
    injury_fn = 0
    death_tp = 0
    death_tn = 0
    death_fp = 0
    death_fn = 0

    for i in range(0, data.shape[0], 32):
        predictions = [tf.squeeze(inj_cls(tf.convert_to_tensor(data.iloc[i:i+32,:]),training = False)), tf.squeeze(death_cls(tf.convert_to_tensor(data.iloc[i:i+32,:]),training = False))]
        predictions = np.round(tf.squeeze(tf.transpose(predictions)))
        for pred, lab in zip(predictions, labels.iloc[i:i+32,:].to_numpy()):

            if (pred[0] == lab[0]) and lab[0] == 0:
                injury_tn += 1
            elif (pred[0] == lab[0]) and lab[0] == 1:
                injury_tp += 1
            elif pred[0] == 0:
                injury_fn += 1
            elif pred[0] == 1:
                injury_fp += 1

            if (pred[1] == lab[1]) and lab[1] == 0:
                death_tn += 1
            elif (pred[1] == lab[1]) and lab[1] == 1:
                death_tp += 1
            elif pred[1] == 0:
                death_fn += 1
            elif pred[1] == 1:
                death_fp += 1
            sample_num += 1

    #Return the number of samples tested, and confusion matrices.
    return sample_num, (injury_tp, injury_fp, injury_tn, injury_fn), (death_tp, death_fp, death_tn, death_fn)

def train(epochs, train_labels, validate_labels, train_data, validate_data):
    for epoch in range(epochs):

        #train once over the dataset.
        for i in range(0, train_data.shape[0], 32):
            printProgressBar(i // 32, train_data.shape[0] // 32, prefix = f'Progress on epoch {epoch}')
            #Seperate the batch data into death / non-death so we can penalize incorrectly classifying death more.
            batch_data = train_data.iloc[i:i+32,:]
            batch_labels = train_labels.iloc[i:i+32,:]
            death_index = batch_labels.index[batch_labels['death_flag'] == 1].tolist()
            death_data = batch_data[batch_data.index.isin(death_index)]
            death_labels = batch_labels[batch_labels.index.isin(death_index)]
            batch_data = batch_data.drop(death_index)
            batch_labels = batch_labels.drop(death_index)
            trainStep(tf.convert_to_tensor(batch_data), tf.transpose(tf.convert_to_tensor(batch_labels), [1,0]), tf.convert_to_tensor(death_data), tf.transpose(tf.convert_to_tensor(death_labels), [1,0]))
            

        #Save the weights of the classifier
        inj_cls.save_weights(CKPT + f"{epoch}inj_cls.ckpt")
        death_cls.save_weights(CKPT + f"{epoch}death_cls.ckpt")

        
        sample_num, injury_cm, death_cm = accuracy(validate_data, validate_labels)
        print(f"Injury Accuracy = {(injury_cm[0]+injury_cm[2])/sample_num}\nDeath Accuracy = {(death_cm[0]+death_cm[2])/sample_num}\n")


    return

def filter_correlation(data):
#Drop the correlated columns, deaths/injuries.
    data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)
    data.drop(list(data.filter(regex="injur")), axis=1, inplace=True)
    data.drop(list(data.filter(regex="death")), axis=1, inplace=True)
    data.drop(list(data.filter(regex="fatal")), axis=1, inplace=True)
    data.drop(list(data.filter(regex="INJUR")), axis=1, inplace=True)
    data.drop(list(data.filter(regex="I_flag")), axis=1, inplace=True)
    return
#Code to check the columns for any missed obviously correlated fields.
# for item in data.columns:
#     print(item)
# exit(1)

#load the data
data = pd.read_csv("Traffic_Crashes_NeuralNet.csv")
data = data.sample(frac=1).reset_index(drop=True) #Shuffle before dividing.
                                                                                # labels = data.filter(regex='_flag')

death_data = data[data['death_flag'] == 1]
non_death_data = data[data['death_flag'] == 0]

#seperate the data into training, testing, and validation sets while maintaining equal ratio of deaths in each set.
death_training = death_data.iloc[:(int)(death_data.shape[0] * 0.7),:]
death_test = death_data.iloc[(int)(death_data.shape[0] * 0.7):(int)(death_data.shape[0] * 0.9),:]
death_validate = death_data.iloc[(int)(death_data.shape[0] * 0.9):,:]

non_death_training = non_death_data.iloc[:(int)(non_death_data.shape[0] * 0.7),:]
non_death_test = non_death_data.iloc[(int)(non_death_data.shape[0] * 0.7):(int)(non_death_data.shape[0] * 0.9),:]
non_death_validate = non_death_data.iloc[(int)(non_death_data.shape[0] * 0.9):,:]

#Construct the dataset with equally distributed deaths in all samples and shuffle after concatenating.
train_data = pd.concat([death_training, non_death_training]).sample(frac=1).reset_index(drop=True)
test_data = pd.concat([death_test, non_death_test]).sample(frac=1).reset_index(drop=True)
validate_data = pd.concat([death_validate, non_death_validate]).sample(frac=1).reset_index(drop=True)

train_labels = train_data.filter(regex='_flag')
test_labels = test_data.filter(regex='_flag')
validate_labels = validate_data.filter(regex='_flag')

filter_correlation(train_data)
filter_correlation(test_data)
filter_correlation(validate_data)

# inj_cls.load_weights(CKPT + f"{4}inj_cls.ckpt")
# death_cls.load_weights(CKPT + f"{4}death_cls.ckpt")

train(5, train_labels, validate_labels, train_data, validate_data)

#find confusion matrices of injuries and deaths.
sample_num, injury_cm, death_cm = accuracy(test_data, test_labels)
print(f"Injury cm: {injury_cm}")
print(f"Death cm: {death_cm}")

#Order for injury/death tuple: (injury_tp, injury_fp, injury_tn, injury_fn)
try:
    injury_precision = injury_cm[0] / (injury_cm[0] + injury_cm[1])
except Exception as e:
    print(f"Error: {e}")
    injury_precision = "N/A"
try:
    death_precision = death_cm[0] / (death_cm[0] + death_cm[1])
except Exception as e:
    print(f"Error: {e}")
    death_precision = "N/A"

try:
    injury_accuracy = (injury_cm[0] + injury_cm[2]) / sample_num
except Exception as e:
    print(f"Error: {e}")
    injury_accuracy = "N/A"
try:
    death_accuracy = (death_cm[0] + death_cm[2]) / sample_num
except Exception as e:
    print(f"Error: {e}")
    death_accuracy = "N/A"

try:
    injury_weighted_acc = test_labels[test_labels['injury_flag'] == 1].shape[0] / sample_num * (injury_cm[0] / (injury_cm[0]+injury_cm[3])) + test_labels[test_labels['injury_flag'] == 0].shape[0] / sample_num * (injury_cm[2] / (injury_cm[1] + injury_cm[2]))
except Exception as e:
    print(f"Error: {e}")
    injury_weighted_acc = "N/A"
try:
    death_weighted_acc = test_labels[test_labels['death_flag'] == 1].shape[0] / sample_num * (death_cm[0] / (death_cm[0]+death_cm[3])) + test_labels[test_labels['death_flag'] == 0].shape[0] / sample_num * (death_cm[2] / (death_cm[1] + death_cm[2]))
except Exception as e:
    print(f"Error: {e}")
    death_weighted_acc = "N/A"

try:
    injury_recall = injury_cm[0] / (injury_cm[0] + injury_cm[3])
except Exception as e:
    print(f"Error: {e}")
    injury_recall = "N/A"
try:
    death_recall = death_cm[0] / (death_cm[0] + death_cm[3])
except Exception as e:
    print(f"Error: {e}")
    death_recall = "N/A"

print(f"Injury Statistics:")
print(f"Precision: {injury_precision}")
print(f"Accuracy: {injury_accuracy}")
print(f"Weighted Accuracy: {injury_weighted_acc}")
print(f"Recall: {injury_recall}")

print(f"Death Statistics:")
print(f"Precision: {death_precision}")
print(f"Accuracy: {death_accuracy}")
print(f"Weighted Accuracy: {death_weighted_acc}")
print(f"Recall: {death_recall}")
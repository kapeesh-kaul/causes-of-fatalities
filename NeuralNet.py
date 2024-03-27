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

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99)

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
    inZ = Input((201), name = 'Input Vector')
    pre = Dense(512)(inZ)
    pre = LeakyReLU(0.2)(pre)

    z = resBlock(pre, 512)
    z = resBlock(z, 512)
    z = Dense(256, activation='sigmoid')(z)

    injury = resBlock(z, 256)
    death = resBlock(z, 256)
    
    injury = Dense(1, name="injury_pred", activation='sigmoid')(injury)
    death = Dense(1, name="fatal_pred", activation='sigmoid')(injury)

    model = tf.keras.Model(inputs=inZ, outputs=[injury, death])
    return model

cls = makeCls()

@tf.function
def trainStep(features, labels):

    with tf.GradientTape() as clsTape:

        predictions = tf.squeeze(cls(features, training=True))

        cls_loss_func = tf.keras.losses.BinaryCrossentropy()

        cls_loss = cls_loss_func(labels, predictions)

    cls_gradients = clsTape.gradient(cls_loss,cls.trainable_variables)

    opt.apply_gradients(zip(cls_gradients,cls.trainable_variables))
    return (cls_loss, predictions)

def train(epochs, train_labels, validate_labels, train_data, validate_data):
    for epoch in range(epochs):

        #train once over the dataset.
        for i in range(0, train_data.shape[0], 32):
            printProgressBar(i // 32, train_data.shape[0] // 32, prefix = f'Progress on epoch {epoch}')
            loss_step, predictions= trainStep(tf.convert_to_tensor(train_data.iloc[i:i+32,:]), tf.transpose(tf.convert_to_tensor(train_labels.iloc[i:i+32,:]), [1,0]))
            

        #Save the weights of the classifier
        cls.save_weights(CKPT + f"{epoch}cls.ckpt")

        #Conduct a validation step to ensure model health.
        def accuracy(data, labels):
            sample_num = 0
            injury_true = 0
            death_true = 0
            for i in range(0, data.shape[0], 32):
                predictions = cls(tf.convert_to_tensor(data.iloc[i:i+32,:]),training = False)
                predictions = np.round(tf.squeeze(tf.transpose(predictions)))
                for pred, lab in zip(predictions, labels.iloc[i:i+32,:].to_numpy()):
                    if pred[0] == lab[0]:
                        injury_true += 1
                    if pred[1] == lab[1]:
                        death_true += 1
                    sample_num += 1
            return injury_true / sample_num, death_true / sample_num
        acc_injure, acc_death = accuracy(validate_data, validate_labels)
        print(f"Injury Accuracy = {acc_injure}\nDeath Accuracy = {acc_death}\n")


    return

#load the data
data = pd.read_csv("Traffic_Crashes_NeuralNet.csv")
labels = data.filter(regex='_flag')

#Drop the correlated columns, deaths/injuries.
data.drop(list(data.filter(regex='Unnamed')), axis=1, inplace=True)
data.drop(list(data.filter(regex="injur")), axis=1, inplace=True)
data.drop(list(data.filter(regex="death")), axis=1, inplace=True)
data.drop(list(data.filter(regex="fatal")), axis=1, inplace=True)

#seperate the data into training, testing, and validation sets.
train_labels = labels.iloc[:(int)(labels.shape[0] * 0.7),:]
test_labels = labels.iloc[(int)(labels.shape[0] * 0.7):(int)(labels.shape[0] * 0.9),:]
validate_labels = labels.iloc[(int)(labels.shape[0] * 0.9):,:]

train_data = data.iloc[:(int)(data.shape[0] * 0.7),:]
test_data = data.iloc[(int)(data.shape[0] * 0.7):(int)(data.shape[0] * 0.9),:]
validate_data = data.iloc[(int)(data.shape[0] * 0.9):,:]

train(5, train_labels, validate_labels, train_data, validate_data)
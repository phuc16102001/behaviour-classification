'''
This is a project for behaviour classification
The project is built to demonstrate for the research methodology course
Owner: Do Vuong Phuc
Reference: Mi AI - Nhan dien hanh vi con nguoi

PLEASE DO NOT COPY WITHOUT PERMISSION!
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from config import *

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

def load_data():
    data_dir = '../data'
    files = os.listdir(data_dir)

    classes = []
    for path in files:
        classes.append(path.split('.')[0])
    list.sort(classes)

    df = {}
    for className in classes:
        df[className] = pd.read_csv(os.path.join(data_dir,className+".csv"))
    return (classes, df)

def encode_data(classes, df):
    nClass = len(classes)

    X = []
    y = []

    for idx, className in enumerate(classes):
        nSample = len(df[className])
        for start in range(nSample-N_TIME):
            X.append(df[className].iloc[start:start+N_TIME,:])
            one_hot = [0]*nClass
            one_hot[idx] = 1
            y.append(one_hot)

    X, y = np.array(X), np.array(y)
    return (X, y)

def get_model(input_shape, classes):
    model = Sequential([
        LSTM(units = 50, return_sequences = True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),
        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),
        LSTM(units = 50),
        Dropout(0.2),
        Dense(units = len(classes), activation='softmax')
    ])

    model.compile(
        optimizer = 'adam',
        metrics = ['accuracy'],
        loss = 'categorical_crossentropy'
    )
    return model

def output_loss(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(20,8))
    epochs = range(1,N_EPOCH+1)

    ax[0].plot(epochs,hist['loss'])
    ax[0].plot(epochs,hist['val_loss'])
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train','Test'])

    ax[1].plot(epochs,hist['accuracy'])
    ax[1].plot(epochs,hist['val_accuracy'])
    ax[1].set_ylabel('Acc (%)')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train','Test'])

    plt.savefig(path)

def main():
    classes, df = load_data()
    X, y = encode_data(classes, df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = get_model(input_shape, classes)
    trainingRet = model.fit(
        X_train, 
        y_train,
        epochs = N_EPOCH,
        batch_size = BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    model.save('../models/best.h5')
    hist = trainingRet.history
    output_loss(hist, '../img/result.png')

if __name__=="__main__":
    main()
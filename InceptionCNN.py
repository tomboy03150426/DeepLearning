# coding: utf-8

import numpy
import pandas as pd
import numpy as np
from time import sleep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,Callback
from time import time
from sklearn import metrics
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Model
import keras

SHAPE = 11

def Feature_Selection(X_Data, Y_Data):
    print("Feature Selection")
    model = ExtraTreesClassifier(n_estimators=500, random_state=0, n_jobs=-1, max_depth=25)
    # n_estimators K值 決策樹數量，越大越好，與學習成本正相關
    # random_state 亂樹種子
    # n_jobs 指定使用多少CPU核心數，-1表示火力全開
    # max_depth 最大深度
    model.fit(X_Data, Y_Data.values.ravel())
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # numpy=np 計算標準差 # axis=軸
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking 印出特徵排名字串
    print("Feature ranking:")
    Best = []
    important = []

    #迴圈印出特徵排名
    for f in range(X_Data.shape[1]):
        if (importances[indices[f]] > 0.000001):
            Best.append(indices[f])
            print("%d. Feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
            important.append(importances[indices[f]])

    print(len(Best), Best)
    X_Data_Best = pd.DataFrame(columns=Best)
    print(X_Data_Best.shape)
    X_Data_Best_copy = pd.DataFrame(columns=Best[:22])
    print(X_Data_Best_copy.shape)
    for i in Best:
        X_Data_Best[i] = X_Data[i]
    print(X_Data_Best.shape)
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_Data_Best_copy.shape[1]), important[:22],
            color="r", yerr=std[Best[:22]], align="center")
    plt.xticks(range(X_Data_Best_copy.shape[1]), Best[:22])
    plt.xlim([-1, X_Data_Best_copy.shape[1]])
    plt.ylabel('Percent')
    plt.xlabel('Feature index')
    plt.show()
    return X_Data_Best

def main():
    print("python main function")
    SIZE_IMG = SHAPE * SHAPE  #11*11
    Train = pd.read_csv("Tor-NonTorAllClean.csv", header=None)

    Y_Data = Train.iloc[:, -1]
    X_Data = Train.iloc[:, :-1]
    print("Printing Y-data")
    print("Printing X-data")
    X_Data_Best = Feature_Selection(X_Data, Y_Data)
    X_Data_Final = Image_Creation(X_Data_Best, SIZE_IMG)
    Y_Data_Final = Label_Encoding(Y_Data)
    MyModel(X_Data_Final, Y_Data_Final)




def Image_Creation(X_Data_Best, SIZE_IMG):
    print("Image Creation")
    i = 0
    X_Data_Final = []
    while i < len(X_Data_Best):
        X = X_Data_Best.iloc[i]
        if SIZE_IMG > len(X):
            X = np.concatenate([X, np.zeros(SIZE_IMG - len(X))])
        X = X.reshape(SHAPE, SHAPE).astype(np.uint8)
        X_Data_Final.append(X)
        i = i + 1
    return X_Data_Final


def Label_Encoding(Y_Data):
    print("Label Encoding")
    Y_Data_Final = Y_Data
    le = LabelEncoder()
    le.fit(Y_Data_Final)
    list(le.classes_)
    Y_Data_Final = le.transform(Y_Data_Final)
    Y_Data_Final = np.reshape(Y_Data_Final, (-1, 1))
    return Y_Data_Final


def MyModel(X_Data_Final, Y_Data_Final):
    print("Model")
    start = time()
    epochs = 50
    optimizer = 'adam'
    numpy.random.seed(9)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X_Data_Final, Y_Data_Final, test_size=0.20, random_state=42)
    X_tr = np.array(X_tr).reshape(-1, SHAPE, SHAPE, 1)
    X_te = np.array(X_te).reshape(-1, SHAPE, SHAPE, 1)
    early_stopping_monitor = EarlyStopping(patience=3)

    print("形狀在此")
    print("X_tr.shape=", X_tr.shape)
    print("X_te.shape=", X_te.shape)
    print("Y_tr.shape=", Y_tr.shape)
    print("Y_te.shape=", Y_te.shape)

    input_img = Input(shape=(11, 11, 1))
    block_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    block_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(block_1)
    block_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    block_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(block_2)
    block_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    block_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(block_3)
    output = keras.layers.concatenate([block_1, block_2, block_3], axis=3)
    output = Flatten()(output)
    out = Dense(2, activation='softmax')(output)
    model = Model(inputs=input_img, outputs=out)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    history = model.fit(X_tr, Y_tr, validation_data=(X_te, Y_te), epochs=epochs, callbacks=[early_stopping_monitor],
                          batch_size=32, verbose=1)
    print(time()-start)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy

    print("神經層數: ", len(model.layers))
    for i in range(len(model.layers)):
        print(i, model.layers[i].name)

    # 模型視覺化存檔
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file="InceptionCNN.png", show_shapes=True)


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    Y_Predict = model.predict_proba(X_te)


######Predicting Category
    i = 0
    y_pred = []
    while i < len(Y_Predict):
        maxi = Y_Predict[i].argmax()
        y_pred.append(maxi)
        i = i + 1
#######Predicting Category over
    print(metrics.classification_report(Y_te, y_pred, digits=2))
    #Please replace the classes in line below with labels in the dataset
    plot_confusion_matrix(Y_te, y_pred, classes = ['Adware','Backdoor','FileInfector','PUA','Ransomware','Riskware','Scareware','Trojan','Trojan_Banker','Trojan_Dropper','Trojan_SMS','Trojan_Spy'],
                          title='Confusion matrix, without normalization')
    #plot_confusion_matrix(Y_te, y_pred, classes = ['BENIGN','DNS','LDAP','MSSQL','NTP','NetBIOS','SNMP','SSDP','Syn','TFTP','UDP','UDPLag'],
     #                    normalize=True, title='Normalized Confusion matrix')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['NonDoh','Doh','Chat','Email','File-Transfer','P2P','Video-Streaming','VOIP']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 14)
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

    return ax


if __name__ == '__main__':
    main()

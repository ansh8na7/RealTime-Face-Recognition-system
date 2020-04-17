# develop a classifier for the Faces Dataset
import os
from numpy import load
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import joblib
# load dataset




def trainModel():
    data = load('faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # print("testX.shape ",testX.shape," type: ",type(testX))

    # print("testy ",testy, " type: ",type(testy))
    # print("testy.shape ",testy.shape)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # print("yhat_test ",yhat_test)
    # print("yhat_test.shape ",yhat_test.shape)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


    # testing single test data
    # x_test = np.asarray([testX[14]])
    # y_test = testy[14]
    # yhat = model.predict(x_test)
    # print(yhat,y_test)

    return model



curr_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(curr_dir,"trained_model.joblib")):
    if input("Tained model exists\n retrain??? (press y = yes): ").lower() == "y":
        FRmodel = trainModel()

        os.remove(os.path.join(curr_dir,"trained_model.joblib"))
        joblib.dump(FRmodel,"trained_model.joblib")

else:
    FRmodel = trainModel()
    joblib.dump(FRmodel,"trained_model.joblib")

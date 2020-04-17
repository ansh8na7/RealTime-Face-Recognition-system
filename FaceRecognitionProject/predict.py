# predict the faces. It is only for trial purpose

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle


FRmodel = joblib.load("trained_model.joblib")

name_label = {}
with open("name_label","rb") as f:
    name_label = pickle.load(f)

data = np.load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

x_test = np.asarray([testX[18]])

in_encoder = Normalizer(norm='l2')
# trainX = in_encoder.transform(trainX)
x_test = in_encoder.transform(x_test)


y_test = testy[14]
yhat = FRmodel.predict(x_test)
print("answer: ",yhat,y_test)
#
inv_dict = {a:b for (b,a) in name_label.items()}
print(inv_dict[yhat[0]],inv_dict[y_test])

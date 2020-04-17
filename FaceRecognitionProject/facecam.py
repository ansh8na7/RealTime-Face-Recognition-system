# use the model to recognize Faces
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from utility import *
from keras.models import load_model
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
# from PIL import Image
import pickle


# load name_label dictionary
name_label = {}
with open("name_label","rb") as f:
    name_label = pickle.load(f)

# inverse of name_label dictionary to map id to name
inv_name_label = {a:b for (b,a) in name_label.items()}

# start capturing
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# load detector and models
detector = MTCNN() #detects faces in image
model = load_model("facenet_keras.h5",compile = False) #Facenet model to encode
FRmodel = joblib.load("trained_model.joblib") #model to classify
in_encoder = Normalizer(norm='l2')

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0,0,255)
stroke = 2


while True:
    ret, frame = cap.read()
    frame = np.asarray(frame)
    faces = detector.detect_faces(frame)

    for face_location in faces:
        x,y,w,h = face_location['box']
        x1,y1 = abs(x),abs(y)
        x2,y2 = x1+w , y1+h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        face_ = Image.fromarray(frame[y1:y2, x1:x2])
        face = face_.resize((160,160))
        face = np.asarray(face)
        embedding = get_embedding(model,face)

        encoding = in_encoder.transform(np.asarray([embedding]))
        prediction = FRmodel.predict(encoding)
        prob = FRmodel.predict_proba(encoding)[0]
        print(prediction)
        print(prob)
        print(max(prob))
        if max(prob)>0.67:
            cv2.putText(frame,inv_name_label[prediction[0]],(x,y),font,1,color,stroke,cv2.LINE_AA)
        else:
            cv2.putText(frame,"???",(x,y),font,1,color,stroke,cv2.LINE_AA)



    cv2.imshow("faces",frame)

    ch = cv2.waitKey(10)
    if ch & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

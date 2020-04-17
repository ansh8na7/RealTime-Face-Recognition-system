# create encoded dataset from the given dataset
import numpy as np
import os
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
import pickle
from utility import * #imports utility functions

# load the name_label dictionary
name_label = {}
with open("name_label","rb") as f:
    name_label = pickle.load(f)

# function to iterate through directories and create X,y dataset for training and testing
def load_dataset(directory):
    Base_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(Base_dir,directory)

    X,y= [],[]

    for root,dirs,files in os.walk(dir):
        key = os.path.basename(root)
        for file in files:
            if not file.startswith("."): #the folder contains .DS_Store, we don't need to consider it in training
                img_path = os.path.join(root,file)
                # print(img_path)     #if no face found in any image, use this print to track that image
                face = extract_face(img_path)
                X.append(face)
                y.append(name_label[key])
    return np.asarray(X),np.asarray(y)




# create training and test datasets
def createDataset():
    trainX,trainy = load_dataset("dataset/train")
    testX,testy = load_dataset("dataset/val")
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

    model = load_model('facenet_keras.h5')

    # convert the x data to encodings
    newTrainX = list()
    for face_pixels in trainX:
    	embedding = get_embedding(model, face_pixels)
    	newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)
    print("newTrainX: ",newTrainX.shape)

    newTestX = list()
    for face_pixels in testX:
    	embedding = get_embedding(model, face_pixels)
    	newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    print("newTestX: ",newTestX.shape)

    # save dataset to disk

    return newTrainX,trainy,newTestX,testy



curr_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(curr_dir,"faces-embeddings.npz")):
    if input("encoded dataset exists\nretrain? (y==yes): ").lower() =="y":
        newTrainX,trainy,newTestX,testy = createDataset()
        os.remove(os.path.join(curr_dir,"faces-embeddings.npz"))
        np.savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

else:
    newTrainX,trainy,newTestX,testy = createDataset()
    np.savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

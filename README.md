## **REALTIME FACE RECOGNITION SYSTEM**

This project is a realtime Face Recognition System using mtcnn, facenet and SVM.

**Install the dependencies using**  $ pip install -r requirements.txt

Add your face dataset to the dataset train and val folder accordingly and run the following scripts in given order:

_**name_label_dictionary.py**_ creates a dictionary allocating ids to their names.

_**create_dataset.py**_ creates the dataset encodings using facenet model.

_**trainer.py**_ trains the SVM model.

_**facecam.py**_ uses all the other models and runs the realtime face recogniser. Once you have trained the model, you only need to run this file to use the facecam and recognise. All other files only need to run if new face is added to dataset.


####this project uses and works on:

tensorflow==2.1.0
tensorflow-addons==0.9.1
Keras==2.3.1
scikit-learn==0.22.2
Pillow==7.1.1
opencv-contrib-python==4.2.0.34
opencv-python==4.2.0.34
numpy==1.18.2
mtcnn==0.1.0

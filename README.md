## **REALTIME FACE RECOGNITION SYSTEM**

This project is a realtime Face Recognition System using mtcnn, facenet and SVM.

Add your face dataset to the dataset train and val folder accordingly and run the following scripts in given order

_name_label_dictionary.py_ creates a dictionary allocating ids to their names

_create_dataset.py_ creates the dataset encodings using facenet model

_trainer.py_ trains the SVM model

_facecam.py_ uses all the other models and runs the realtime face recogniser


#### this project uses and works on:

tensorflow==2.1.0

tensorflow-addons==0.9.1

Keras==2.3.1

scikit-learn==0.22.2

Pillow==7.1.1

opencv-contrib-python==4.2.0.34

opencv-python==4.2.0.34

numpy==1.18.2

mtcnn==0.1.0

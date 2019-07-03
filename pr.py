
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# initialising the cnn
classifier =Sequential()
# step -1 = convolution
classifier.add(Convolution2D(32,3,3,input_shape=(56,56,3),activation='relu'))
# step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
# step3 flattening
classifier.add(Flatten())
# step 4 full connection
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="sigmoid", units=1))
# fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
# compiling the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# training set
training_set = train_datagen.flow_from_directory("flowers",target_size=(56,56),batch_size=32,class_mode='binary')
test_set = train_datagen.flow_from_directory("flowers" ,target_size=(56,56),batch_size=32,class_mode='binary')
classifier.fit_generator(training_set,samples_per_epoch=6072,nb_epoch=10,validation_data=training_set,nb_val_samples=1518)
import cv2
cap=cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
    cv2.imshow('frame',rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out=cv2.imwrite("flowers.jpg",frame)
        break
cap.release()
cv2.destroyAllWindows()    
from keras.preprocessing  import image
import numpy as np
test_set = image.load_img("flowers.jpg",target_size=(56,56))
#test_set = test_set.reshape((-1,56,56,3))
test_set=np.expand_dims(test_set,axis=0)
y_pred=classifier.predict([test_set])

if y_pred[0][0] == 0:
    print("rose")
elif   y_pred[0][0] == 1:
    print("sunflower") 
else:
    print("NO FLOWER EXIST")
    
    



  

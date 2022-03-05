# AlexNet and Our Own classifier

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import datetime
import cv2
import matplotlib.pyplot as plt


# printout versions
print(f"Tensor Flow Version: {tf.__version__}")
print(f"numpy Version: {np.version.version}")

data_dir = pathlib.Path("/content/drive/My Drive/ML_Project/Train")
image_count = len(list(data_dir.glob('*/*.JPG')))
print(image_count)
# classnames in the dataset specified
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" ])
print(CLASS_NAMES)
# print length of class names
output_class_units = len(CLASS_NAMES)
print(output_class_units)

model = tf.keras.models.Sequential([
    # 1st conv
  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3rd conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4th conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5th Conv
  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  tf.keras.layers.Flatten(),
  # To FC layer 1
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  #To FC layer 2
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(output_class_units, activation='softmax')
])

BATCH_SIZE = 32             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 227            # input Shape required by the model
IMG_WIDTH = 227             # input Shape required by the model
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

# Rescalingthe pixel values from 0~255 to 0~1 For RGB Channels of the image.
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# training_data for model training
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))
                                                     
 
model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

# Summarizing the model architecture and printing it out
model.summary()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get("accuracy")==1.00 and logs.get("loss")<0.03):
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training =True
callbacks = myCallback()

# TensorBoard.dev Visuals
log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
      train_data_gen,
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=50,
      callbacks=[tensorboard_callback,callbacks])

# Saving the model
model.save('/content/drive/My Drive/ML_Project/Models/{NAME}.model')


# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()

import tensorflow as tf
import numpy as np
import pathlib
import datetime
# Raw Dataset Directory
data_dir = pathlib.Path("/content/drive/My Drive/ML_Project/Test")
image_count = len(list(data_dir.glob('*/*.JPG')))
# print total no of images for all classes
print(image_count)
# classnames in the dataset specified
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt" ])
# print list of all classes
print(CLASS_NAMES)
# print length of class names
output_class_units = len(CLASS_NAMES)
print(output_class_units)
#preprocess the data
BATCH_SIZE = 1             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 227            # input Shape required by the model
IMG_WIDTH = 227             # input Shape required by the model

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES))
#Loading the saved model
new_model = tf.keras.models.load_model("/content/drive/My Drive/ML_Project/Models/{NAME}.model")
new_model.summary()
loss, acc = new_model.evaluate(test_data_gen)
print("accuracy:{:.2f}%".format(acc*100))


# Our Own Classifier

import cv2
import numpy as np
import os
import csv

trainingDataset = []
img_size = 100
path = "/content/drive/My Drive/ML_Project2/Train"

classNumber = 0
trainingDataset.clear()

for folder in (os.listdir(path)):
  print(classNumber)
  print("Folder Name:",folder)
  # folder = with_mask ,without_mask
  fp = os.path.join(path,folder)
  # joining folder like /content/Face_Mask/Train/with_mask
  for eachImage in os.listdir(fp):
    imagePath = os.path.join(fp,eachImage)
    img = (cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE))/255
    resize=cv2.resize(img,(img_size,img_size))
    trainingDataset.append([resize,classNumber])
  classNumber = classNumber + 1
  
  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import time

X = []
Y = []
img_size = 100
np.random.shuffle(trainingDataset)
for features, label in trainingDataset:
    X.append(features)
    Y.append(label)
print(Y) 

X = np.array(X).reshape(-1, img_size, img_size, 1)
Y_binary = to_categorical(Y)
print(Y_binary)

model = Sequential()

model.add(Conv2D(200, (3, 3), input_shape=(100,100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
 
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
              
history = model.fit(X, Y_binary,
          batch_size = 32,
          epochs=20, validation_split = 0.1)
 
model.save("/content/drive/My Drive/ML_Project2/Models/{NAME}.model")


plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()

def prepare(filepath):
    img_size = 100 
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255  
    img_resize = cv2.resize(img, (img_size, img_size))  
    return img_resize.reshape(-1, img_size, img_size, 1)
 
 
prediction = model.predict(prepare("/content/drive/My Drive/ML_Project2/Test/Pepper_Unhealthy/check.JPG"))
print((prediction))

CATEGORIES = ["healthy", "unhealthy"]

pred_class = CATEGORIES[np.argmax(prediction)]
print(pred_class)

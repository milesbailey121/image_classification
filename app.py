#https://www.youtube.com/watch?v=jztwpsIzEGc 
#Youtube tutorial

import tensorflow as tf
import os 
import numpy as np
from matplotlib import pyplot as plt

#
# Step 1: Loading Dataset 
#

gpus = tf.config.experimental.list_physical_devices("GPU") # Lists all physical GPU's and limits the amount of VRAM accesable to tf
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    
import cv2 # computer vision
import imghdr # check file extension 


data_dirct = "data"
image_exts = ["jpeg","jpg","bmp","png"]

def image_cleaning():
    for image_class in os.listdir(data_dirct): # Loops through every folder
        for image in os.listdir(os.path.join(data_dirct, image_class)): # Loops through every file
            image_path = os.path.join(data_dirct, image_class, image) # creates a file path for each file
            try:
                img = cv2.imread(image_path) # reads if it's possible to be opened by cv2
                tip = imghdr.what(image_path) # reads if incorrect file format
                if tip not in image_exts:
                    print("Image not in ext list {}" .format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print("Issues with image {}".format(image_path))
                #os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory("data") # generates a compatable dataset and formats corretly
data_iterator = data.as_numpy_iterator() # creates it into a referencable numpy arrays
batch = data_iterator.next() # gets a group of images from the iterator

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])      
#plt.show()

#
# Step 2: Preprocessing Dataset 
#

#Scale dataset

data = data.map(lambda x,y: (x/255, y)) # allows us to perform a function during pipelin, x = image y=labels
scaled_iterator = data.as_numpy_iterator()
print(scaled_iterator.next())
print(scaled_iterator.next()[0].max())
print(scaled_iterator.next()[0].min())


#Split dataset

train_size = int(len(data)*.7) 
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Deep learning

import tensorflow as tf; tf.keras.models 

#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(16,(3,3),1,activation="relu",input_shape=(256,256,3))) # Generates a conv 16 filter of (3,3) and stide of 1 through a relu activation 
model.add(tf.keras.layers.MaxPooling2D()) # Gets the max value out of a 2 by 2 

model.add(tf.keras.layers.Conv2D(32,(3,3),1,activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(16,(3,3),1,activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])


#Train model

logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])

#plot performance 

fig = plt.figure()
plt.plot(history.history["loss"],color = "teal", label="loss")
plt.plot(history.history["val_loss"],color = "orange", label="val_loss")
fig.suptitle("Loss",fontsize=20)
plt.legend(loc="upper left")
plt.show

fig = plt.figure()
plt.plot(history.history["accuracy"],color = "teal", label="accuracy")
plt.plot(history.history["val_accuracy"],color = "orange", label="val_accuracy")
fig.suptitle("accuracy",fontsize=20)
plt.legend(loc="upper left")
plt.show

#evaulate performance
pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y,yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat)
    
print(f"Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")


#TESTING ON NEW DATA
img = cv2.imread("data\happytest.png")
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255,0))


img = cv2.imread("data\sadtest.png")
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255,0))


#Saving and loading models
model.save(filepath=(os.path.join("models","imageclassifier.h5")))
new_model = tf.keras.models.load_model(os.path.join("models","imageclassifier.h5"))
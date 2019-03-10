# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:04:37 2018
#* ()_":
@author: Gulshan Rana
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
from skimage.color import *   
from tensorflow.keras.layers import *   
import tensorboard
X=[]
for filename in os.listdir('Train/'):
    X.append(img_to_array(load_img('Train/'+filename)))  
X = np.array(X, dtype=float)
split = int(0.95*len(X)) 
Xtrain= X[:split]
Xtrain= 1.0/255*Xtrain

model= tf.keras.Sequential()
model.add(InputLayer(input_shape=(256,256,1)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same',strides=2))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same',strides=2))
model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
model.add(Conv2D(256,(3,3),activation='relu',padding='same',strides=2))
model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
model.add(Conv2D(256,(3, 3), activation='relu', padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(2,(3,3),activation='tanh',padding='same'))


model.compile(loss='mse',optimizer='rmsprop')

datagen=ImageDataGenerator(shear_range=0.2,horizontal_flip=True,zoom_range=0.2,rotation_range=20)

batch_size= 50
def image_ab_gen(batch_size):
    for batch in datagen.flow(Xtrain,batch_size=batch_size):
        lab_batch= rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
        
#tensorboard(log_dir='/output')
model.fit_generator(image_ab_gen(batch_size),epochs=50,steps_per_epoch=10)

Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest=Ytest/128
print (model.evaluate(Xtest, Ytest, batch_size=batch_size))
#* ()_":
color_me = []
for filename in os.listdir('Test/'):
        color_me.append(img_to_array(load_img('Test/'+filename)))
color_me= np.array(color_me,dtype= float)
color_me=((1.0/255)* color_me[:,:,:,0]) 
color_me= color_me.reshape(color_me.shape+(1,))

output= model.predict(color_me)
output= output*128

for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("img_result"+str(i)+".png", lab2rgb(cur))  
    




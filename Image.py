import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

import random

import tensorflow as tf
from tensorflow  import keras
from tensorflow.keras import layers




# print(tf.__version__)


# img_array = cv2.imread("C:/Users/ASUS/Music/SE Semester Project/training/0/Training_3908.jpg")
# print(img_array.shape)
# plt.imshow(img_array)
# plt.show()  
#print(img_array)

Datadirectory="C:/Users/ASUS/Music/SE Semester Project/train/"
Classes=["0","1","2","3","4","5","6"]
for category in Classes:
    path=os.path.join(Datadirectory,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        # plt.show()
        break
    break
img_size=224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
# plt.show()
#print(new_array.shape)

training_Data=[]

      
def create_training_Data():
      for category in Classes:
            path=os.path.join(Datadirectory,category)
            class_num=Classes.index(category)
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img))
                    new_array=cv2.resize(img_array,(img_size,img_size))
                    training_Data.append([new_array,class_num])
                except Exception as e:
                    pass
create_training_Data()
#print(len(training_Data))
# temp=np.array(training_Data)
# print(temp.shape)
# zimport random
random.shuffle(training_Data)

x=[]
y=[]

for features,label in training_Data:
    x.append(features)
    y.append(label)
x=np.array(x).reshape(-1,img_size,img_size,3).astype('float16')
x=x/255.0;

# print(x.shape)   

# print(y[0])

# print(type(y))

Y=np.array(y)

# print(Y.shape)


# model = tf.keras.applications.MobileNetV2()

# # model.summary()

# base_input=model.layers[0].input

# base_output=model.layers[-2].output

# # print(base_output)

# final_output = layers.Dense(128)(base_output)
# final_output = layers.Activation('relu')(final_output) 
# final_output = layers.Dense(64)(final_output)
# final_output = layers.Activation('relu')(final_output)
# final_output = layers.Dense(7, activation='softmax')(final_output)      

# # print(final_output)

# new_model = keras.Model(inputs = base_input, outputs = final_output)

# new_model.summary()











model = tf.keras.applications.MobileNetV2()


base_input = model.input
base_output = model.layers[-2].output 


final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output) 
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)  


new_model = keras.Model(inputs=base_input, outputs=final_output)




new_model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


new_model.fit(x,Y,epochs=25)
new_model.save('Final_model_hamza.h5')
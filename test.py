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

# tf.config.set_visible_devices([], 'GPU')

new_model=tf.keras.models.load_model('C:/Users/ASUS/Music/SE Semester Project/Final_model_95p07.h5')
# print(tf.__version__) 

# new_model.evaluate

frame=cv2.imread("surprise.jpg")
print(frame.shape)  
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
# plt.show()   

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
print(gray.shape)

faces=faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+h,x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)   
    facess=faceCascade.detectMultiScale(roi_gray)
    if len(faces)==0:
        print("Face not detected")
    else:
        for(ex,ey,ew,eh) in facess:
            face_roi=roi_color[ey:ey+eh,ex:ex+ew]


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
plt.show()   

# plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
# plt.show()  


final_image=cv2.resize(face_roi,(224,224))
final_image=np.expand_dims(final_image,axis=0)
final_image=final_image/255.0

Predictions=new_model.predict(final_image)

print(Predictions[0])
print(np.argmax(Predictions)) 
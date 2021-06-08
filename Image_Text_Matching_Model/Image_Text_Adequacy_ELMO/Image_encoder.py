#!/usr/bin/env python
# coding: utf-8

# In[28]:


from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras import Model
from keras.preprocessing import image as kimage
import numpy as np


# In[29]:


def load_Encoder():
    # Load the inception v3 model
    model_encoder = InceptionV3(weights='imagenet')
    # Deffining the Encoder model
    model_encoder = Model(model_encoder.input, model_encoder.layers[-2].output)
    return model_encoder

def preprocess_img(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = kimage.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = kimage.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x
    
# Function to encode a given image into a vector of size (2048, )
def encode(image_path, model):
    image = preprocess_img(image_path) # preprocess the image
    fea_vec = model.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


# In[30]:


model = load_Encoder()
encoded_img = encode("E:/TFM/Flickr8k/Images/667626_18933d713e.jpg",model)


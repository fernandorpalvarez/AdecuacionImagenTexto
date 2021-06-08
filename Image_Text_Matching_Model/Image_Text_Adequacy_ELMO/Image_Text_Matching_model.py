#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import Model
from keras.preprocessing import image as kimage
import pickle

# # TEXT ENCODING
import numpy as np
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    
    i=0
    
    descriptions["id"] = list()
    descriptions["label"] = list()
    descriptions["image_id"] = list()
    descriptions["desc"] = list()
    
    # As the dataset is not prepared for a multimodal binary classification
    # let's mix some instances and make the label '0' means that the image 
    # and the text are similar and the laber '1' means that are not similar
    lines = doc.split('\n')
    total_instances = len(dataset)
    positivos = int(0.8 * total_instances)
    negativos = total_instances - positivos

    for j, line in enumerate(lines):        
        # split line by white space
        tokens = line.split()
        
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        
        desc = ""
        for w in image_desc:
            desc = desc + " " + w
            
        if image_id in dataset:
            
            # create list
            descriptions["id"].append(i)
            descriptions["desc"].append(desc)
            
            if i<positivos*5:
                descriptions["image_id"].append(image_id)
                descriptions["label"].append(0)

            else:
                original_state = j
                j = j-10
                tokens = lines[j].split()
                
                # split id from description
                image_id, image_desc = tokens[0], tokens[1:]
                
                while image_id not in dataset:
                    j = j-5
                    tokens = lines[j].split()
                    # split id from description
                    image_id, image_desc = tokens[0], tokens[1:]
                
                descriptions["image_id"].append(image_id)
                descriptions["label"].append(1)  
                    
                j = original_state
                
            i+=1

    return descriptions

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

encoder = load_Encoder()

# Function to encode a given image into a vector of size (2048, )
def encode(image_path, model=encoder):
    image = preprocess_img(image_path)  # preprocess the image
    fea_vec = model.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
    return fea_vec


def add_images_encoded(df, features):
    img_encoded = list()
    for n in df['image_id']:
        name = n + ".jpg"
        img_encoded.append(features[name])

    df['image_encoded'] = img_encoded


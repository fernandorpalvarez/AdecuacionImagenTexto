# img_viewer.py

import base64
import io
import os
import pickle
import numpy as np
import PySimpleGUI as sg
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image, ImageTk
import Image_Text_Matching_Model.Image_Text_Adequacy_ELMO.Image_Text_Matching_model as image_encoder
import Image_Text_Matching_Model.Image_Text_Adequacy_ELMO.Text_encoder as text_encoder


# ------------------------------------------------------------------------------
# use PIL to read data of one image
# ------------------------------------------------------------------------------


def get_img_data(f, maxsize=(1200, 850), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    if img.format == "GIF":
        print("GIF")
        contents = open(os.path.join( f), 'rb').read()
        encoded = base64.b64encode(contents)
        return encoded
    img.thumbnail(maxsize)
    if first:  # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)


# ------------------------------------------------------------------------------



# load the model from disk
lreg = pickle.load(open("C:/Users/rubio/Jupyter Notebook/Adecuacion Imagen Texto/Image_Text_Matching_Model/Image_Text_Adequacy_ELMO/logistic_regression.sav", 'rb'))
try:
    sequential = tf.keras.models.load_model("C:/Users/rubio/Jupyter Notebook/Adecuacion Imagen Texto/Image_Text_Matching_Model/"
                                            "Image_Text_Adequacy_ELMO/sequential_models/sequential.h5")
except:
    print('hola')
elmo = hub.load("https://tfhub.dev/google/elmo/2")
img_encoder = image_encoder.load_Encoder()

# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Directorio:"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
         sg.Text('Introduce un texto:')
    ],
    [
         sg.Input(do_not_clear=True, key="texto"),
         sg.Button('Calcular similitud', key="-SIMILARITY-")
    ],
    [
         sg.Text('Aquí se mostrará el resultado',key="-resultado-")
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Escoge una imagen del directorio izquierdo:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(data=None, enable_events=True,key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image-Text Adequacy", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".jpg", ".gif",".png", ".jpeg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = values["-FOLDER-"] + "/" + values["-FILE LIST-"][0]
            image = get_img_data(filename, first=True)
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(data=image)

        except:
            pass

    elif event == "-SIMILARITY-": # Buttom

        try:
            # get the image and encode it
            filename = values["-FOLDER-"] + "/" + values["-FILE LIST-"][0]
            img_encoded = image_encoder.encode(filename,img_encoder)

            # get the text
            text = values["texto"]
            text_encoded = text_encoder.encode([text],elmo)[0][0].numpy()

            # Concatenate the vectors
            input = np.asarray([np.concatenate((text_encoded, img_encoded))])
            output = sequential.predict_classes(input)[0][0]

            if output == 0:
                texto = "Sí hay similitud"
                print(texto)
                window["-resultado-"].update(value = texto)
            else:
                texto = "No hay similitud"
                print(texto)
                window["-resultado-"].update(value = texto)

        except Exception as e:
            print(e)

window.close()
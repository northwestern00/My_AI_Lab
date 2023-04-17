"""
Created on Mon Apr 17 09:39:55 2023

@author: Özgür Özcan
"""

#GUI Libraries
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from tkinter.ttk import Progressbar

#Data Processing Libraries
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

#CNN Libraries with Keras and Tensorflow
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import load_model
from keras.optimizers import Adam

#Define the metadata to our algorythm
skin_dataframe = pd.read_csv("HAM10000_metadata.csv")

#Control Variables in metadata
skin_dataframe.head()
skin_dataframe.info()
sns.countplot(x = "dx", data = skin_dataframe)

#There is no errors to our metadata, we can get preprocess metadata for use in DeepLearning
#Define a variable name for our image folder which we use
data_folder_name = "HAM10000_images_part_1/"
ext = ".jpg"
skin_dataframe["path"] = [data_folder_name + i + ext for i in skin_dataframe["image_id"]]
skin_dataframe["image"] = skin_dataframe["path"].map( lambda x: np.asarray(Image.open(x).resize((100,75))))
skin_dataframe["dx_idx"] = pd.Categorical(skin_dataframe["dx"]).codes

#Let's standardize to our data frame with OneHotEncoding
x_train = np.asarray(skin_dataframe["image"].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean)/x_train_std

y_train = to_categorical(skin_dataframe["dx_idx"], num_classes = 7)


model1 = load_model("deep_model1.h5")
model2 = load_model("deep_model2.h5")

#index = 5
#y_pred = model1.predict(x_train[index].reshape(1,75,100,3))
#y_pred_class = np.argmax(y_pred, axis = 1)
#Classification Program GUI

window = tk.Tk()
window.geometry("1080x640")
window.wm_title("GUNAM Artifical Intelligence Lab Analyzer(Developed By Özgür Özcan)")


#Global variables
img_name = ""
count = 0
img_jpg = ""
frame_left = tk.Frame(window, width = 540, height = 640, bd = "2")
frame_left.grid(row = 0, column = 0)

frame_right = tk.Frame(window, width = 540, height = 640, bd = "2")
frame_right.grid(row = 0, column = 1)

frame1 = tk.LabelFrame(frame_left, text = "Image", width = 540, height = 500)
frame1.grid(row = 0, column = 0)

frame2 = tk.LabelFrame(frame_left, text = "Core Selection and Save", width = 540, height = 140)
frame2.grid(row = 1, column = 0)

frame3 = tk.LabelFrame(frame_right, text = "Features", width = 270, height = 640)
frame3.grid(row = 0, column = 0)

frame4 = tk.LabelFrame(frame_right, text = "Results", width = 270, height = 640)
frame4.grid(row = 0, column = 1, padx = 10)

#Fill the Frames
def imageResize(img):
    basewidth = 500
    wpercent = (basewidth/float(img.size[0])) #1000*1200
    hsize = int ((float(img.size[1]*float(wpercent))))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS) 
    return img

def openImage():
    global img_name
    global count
    global img_jpg
    count +=1
    if count !=1:
        messagebox.showinfo(title = "Warning", message = "Only one image can be opened")
    else:
       img_name = filedialog.askopenfilename(initialdir = "C:/Users/unalm/Downloads/archive (1)", title = "Select an image file")
       img_jpg = img_name.split("/")[-1].split(".")[0]
       #image label
       tk.Label(frame1, text = img_jpg, bd = 3).pack(pady = 10)
       #open and show image
       img = Image.open(img_name)
       img = imageResize(img)
       img = ImageTk.PhotoImage(img)
       panel = tk.Label(frame1, image = img)
       panel.image = img
       panel.pack(padx = 15, pady = 10)
       #Image Feature
       data = pd.read_csv("HAM10000_metadata.csv")
       cancer = data[data.image_id == img_jpg]
       for i in range(cancer.size):
           x = 0.4
           y = (i/10)/2
           tk.Label(frame3, font = ("Times", 12), text = str(cancer.iloc[0,i])).place(relx = x, rely = y)
       
menubar = tk.Menu(window)
window.config(menu = menubar)
file = tk.Menu(menubar)
menubar.add_cascade(label ="file", menu = file)
file.add_command(labe = "Open", command = openImage)

#Frame-3
def classification():
    if img_name !="" and models.get() !="":
        if models.get() == "Core L":
            #Core Selection
            classification_model = model1
        else:
            classification_model = model2
        z = skin_dataframe[skin_dataframe.image_id == img_jpg]
        z = z.image.values[0].reshape(1,75,100,3)
        z = (z - x_train_mean)/x_train_std
        h = classification_model.predict(z)[0]
        h_index = np.argmax(h)
        predicted_cancer = list(skin_dataframe.dx.unique())[h_index]
        
        for i in range(len(h)):
            x = 0.5
            y = (i/10)/2
            
            if i !=h_index:
                tk.Label(frame4, text = str(h[i])).place(relx = x , rely = y)
            else:
                tk.Label(frame4, bg = "green", text = str(h[i])).place(relx = x, rely = y)
        
        
columns = ["Material ID", "Image ID", "DX", "DX Type", "Scope", "Material Type", "Roughness"]

for i in range(len(columns)):
    x = 0.1
    y = (i/10)/2
    tk.Label(frame3, font = ("Times", 8), text = str(columns[i])+ ":").place(relx = x, rely = y)

classify_button = tk.Button(frame3, bg = "white", bd = 4, font = ("Times", 13), activebackground = "gray", text = "Analyze Data", command = classification)
classify_button.place(relx = 0.1, rely = 0.5)

#Frame 4
labels = skin_dataframe.dx.unique()
for i in range(len(columns)):
    x = 0.1
    y = (i/10)/2
    tk.Label(frame4, font = ("Times", 12), text = str(labels[i]) + ":").place(relx = x, rely = y)

#Frame 2 
#Combo Box

model_selection_label = tk.Label(frame2, text  ="Deep Learning Core")
model_selection_label.grid(row = 0, column = 0, padx = 5)

models = tk.StringVar()
model_selection = ttk.Combobox(frame2, textvariable = models, values = ("Core L", "Core H"), state = "readonly")
model_selection.grid(row = 0, column = 1, padx = 5)

#Check Box
chvar = tk.IntVar()
chvar.set(0)
xbox = tk.Checkbutton(frame2, text = "Save Analysis Result", variable = chvar)
xbox.grid(row = 1, column = 0, pady = 5)

#Entry
entry = tk.Entry(frame2, width = 23)
entry.insert(string = "Saving name...", index = 0)
entry.grid(row = 1, column = 1)




window.mainloop()
